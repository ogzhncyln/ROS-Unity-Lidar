using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;
using UnitySensors.ROS.Publisher;

public class LidarCPU : MonoBehaviour
{
    public float freq=10f;
    public int num_horizontal_scans = 1800;
    public int num_vertical_scans = 16;
    public float vertical_fov = 30f;
    public float range = 40f;
    public string output_topic = "/velodyne_points";
    public string frame_id = "velodyne";

    private NativeArray<RaycastCommand> ray_cmds;
    private NativeArray<RaycastHit> hits;
    private ROSConnection ros;


    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<PointCloud2Msg>(output_topic,1);

        ray_cmds = new NativeArray<RaycastCommand>(num_horizontal_scans * num_vertical_scans,Allocator.Persistent);
        hits = new NativeArray<RaycastHit>(num_horizontal_scans * num_vertical_scans,Allocator.Persistent);
        StartCoroutine(Timer(1f / freq));
    }

    IEnumerator Timer(float t)
    {
        while (true)
        {
            yield return new WaitForSeconds(t);
            Scan();
        }
    }

    [BurstCompile]
    public struct CalculateDirectionJob : IJobParallelFor
    {
        public int num_horizontal_scans;
        public int num_vertical_scans;
        public float vertical_fov;
        public float range;
        public NativeArray<RaycastCommand> ray_cmds;
        public Vector3 current_position;
        public Vector3 forward;
        public Vector3 right;
        public Vector3 up;

        public void Execute(int index)
        {
            int horizontal_index = Mathf.FloorToInt(index / num_vertical_scans);
            int vertical_index = index % num_vertical_scans;

            float horizontal_angle = 2f * Mathf.PI * horizontal_index / num_horizontal_scans;
            float vertical_angle = vertical_fov / 2f - vertical_index * vertical_fov / num_vertical_scans;

            ray_cmds[index] = new RaycastCommand(
                current_position,
                (right * Mathf.Cos(horizontal_angle) + up * Mathf.Sin(vertical_angle) + forward * Mathf.Sin(horizontal_angle)).normalized,
                QueryParameters.Default,
                range
            );
        }
    }

    [BurstCompile]
    public struct HandleSensorDataJob : IJobParallelFor
    {
        public Vector3 pos;
        public Quaternion rot;
        public NativeArray<RaycastHit> hits;
        public NativeArray<float> x,y,z,time;
        public NativeArray<ushort> ring;
        public int num_vertical_scans;

        public void Execute(int index)
        {
            Vector3 p = hits[index].point - pos;
            Vector3 local = Quaternion.Inverse(rot) * p;

            x[index] = local.x;
            y[index] = local.y;
            z[index] = local.z;
            ring[index] = Convert.ToUInt16(index % num_vertical_scans);
            time[index] = 0f;
        }
    }

    private void Scan()
    {
        JobHandle handle;

        CalculateDirectionJob direction_job = new CalculateDirectionJob()
        {
            num_horizontal_scans = num_horizontal_scans,
            num_vertical_scans = num_vertical_scans,
            vertical_fov = vertical_fov * Mathf.Deg2Rad,
            range = range,
            current_position = transform.position,
            forward = transform.forward,
            right = transform.right,
            up = transform.up,
            ray_cmds = ray_cmds
        };

        handle = direction_job.Schedule(ray_cmds.Length,1);
        handle.Complete();

        handle = RaycastCommand.ScheduleBatch(ray_cmds,hits,1);
        handle.Complete();

        List<RaycastHit> hits_list = new List<RaycastHit>();
        List<float> intensity_list = new List<float>();

        foreach(RaycastHit hit in hits)
        {
            if(hit.collider != null)
            {
                hits_list.Add(hit);
                intensity_list.Add(hit.collider.GetComponent<Renderer>() != null ? (float)hit.collider.GetComponent<Renderer>().sharedMaterial.color.grayscale*65536f : (float)0);
            }
        }

        NativeArray<RaycastHit> hits_filtered = new NativeArray<RaycastHit>(hits_list.ToArray(),Allocator.TempJob);

        NativeArray<float> x = new NativeArray<float>(hits_filtered.Length,Allocator.TempJob);
        NativeArray<float> y = new NativeArray<float>(hits_filtered.Length,Allocator.TempJob);
        NativeArray<float> z = new NativeArray<float>(hits_filtered.Length,Allocator.TempJob);
        NativeArray<float> intensity = new NativeArray<float>(intensity_list.ToArray(),Allocator.TempJob);
        NativeArray<float> time = new NativeArray<float>(hits_filtered.Length,Allocator.TempJob);
        NativeArray<ushort> ring = new NativeArray<ushort>(hits_filtered.Length,Allocator.TempJob);

        HandleSensorDataJob handle_job = new HandleSensorDataJob()
        {
            pos = transform.position,
            rot = transform.rotation,
            x = x,
            y = y,
            z = z,
            time = time,
            ring = ring,
            num_vertical_scans = num_vertical_scans,
            hits = hits_filtered
        };

        handle = handle_job.Schedule(hits_filtered.Length,1);
        handle.Complete();

        int pointCount = ring.Length;

        PointCloud2Msg pointCloudMsg = new PointCloud2Msg
        {
            header = new HeaderMsg
            {
                stamp = new RosMessageTypes.BuiltinInterfaces.TimeMsg((uint)Time.time, (uint)((Time.time - (int)Time.time) * 1e9)),
                frame_id = frame_id
            },
            height = 1,
            width = (uint)pointCount,
            fields = new[]
            {
                new PointFieldMsg { name = "x", offset = 0, datatype = 7, count = 1 },
                new PointFieldMsg { name = "y", offset = 4, datatype = 7, count = 1 },
                new PointFieldMsg { name = "z", offset = 8, datatype = 7, count = 1 },
                new PointFieldMsg { name = "intensity", offset = 12, datatype = 7, count = 1 },
                new PointFieldMsg { name = "ring", offset = 16, datatype = 4, count = 1 },
                new PointFieldMsg { name = "time", offset = 18, datatype = 7, count = 1 }
            },
            is_bigendian = false,
            point_step = 22,
            row_step = (uint)(22 * pointCount),
            is_dense = true,
            data = new byte[22 * pointCount]
        };

        for (int i = 0; i < pointCount; i++)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(x[i]), 0, pointCloudMsg.data, i * 22, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(z[i]), 0, pointCloudMsg.data, i * 22 + 4, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(y[i]), 0, pointCloudMsg.data, i * 22 + 8, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(intensity[i]), 0, pointCloudMsg.data, i * 22 + 12, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(ring[i]), 0, pointCloudMsg.data, i * 22 + 16, 2);
            Buffer.BlockCopy(BitConverter.GetBytes(time[i]), 0, pointCloudMsg.data, i * 22 + 18, 4);
        }

        ros.Publish(output_topic, pointCloudMsg);

        x.Dispose();
        y.Dispose();
        z.Dispose();
        intensity.Dispose();
        time.Dispose();
        ring.Dispose();
    }

    void OnDestroy()
    {
        if (ray_cmds.IsCreated) ray_cmds.Dispose();
        if (hits.IsCreated) hits.Dispose();
    }
}
