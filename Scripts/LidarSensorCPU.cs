using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector;
using System;
using RosMessageTypes.Std;
using UnityEngine.XR;
using UnitySensors.ROS.Publisher;
using System.Linq;
using Unity.VisualScripting;

public class LidarSensorCPU : MonoBehaviour
{
    public float freq=10f;
    public int num_horizontal_scans = 1800;
    public int num_vertical_scans = 16;
    public float vertical_fov = 30f;
    public int num_parts = 6;

    private int part_size;
    private int part_index;

    private NativeArray<RaycastCommand>[] rays;
    private NativeArray<RaycastHit>[] hits;
    private NativeArray<float> x,y,z,i;
    private NativeArray<ushort> r;

    private float[] time;

    private ROSConnection ros;  


    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<PointCloud2Msg>("/velodyne_points",1);

        if(num_horizontal_scans % num_parts == 0)
            part_size = Mathf.FloorToInt(num_horizontal_scans / num_parts);
        else
        {
            Debug.LogError("num_horizontal_scans % num_parts != 0");
            return;
        }

        rays = new NativeArray<RaycastCommand>[num_parts];
        hits = new NativeArray<RaycastHit>[num_parts];

        x = new NativeArray<float>(num_horizontal_scans * num_vertical_scans,Allocator.TempJob);
        y = new NativeArray<float>(num_horizontal_scans * num_vertical_scans,Allocator.TempJob);
        z = new NativeArray<float>(num_horizontal_scans * num_vertical_scans,Allocator.TempJob);
        i = new NativeArray<float>(num_horizontal_scans * num_vertical_scans,Allocator.TempJob);
        r = new NativeArray<ushort>(num_horizontal_scans * num_vertical_scans,Allocator.TempJob);

        for(int i = 0; i < num_parts; i++)
        {
            rays[i] = new NativeArray<RaycastCommand>(part_size*num_vertical_scans,Allocator.TempJob);
            hits[i] = new NativeArray<RaycastHit>(part_size*num_vertical_scans,Allocator.TempJob);
        }

        float dt = 1f / freq;

        time = new float[num_horizontal_scans * num_vertical_scans];

        for(int i = 0; i < num_horizontal_scans * num_vertical_scans; i++)
        {
            int index = Mathf.FloorToInt(i / num_vertical_scans);
            float val = dt / num_horizontal_scans * index;
            time[i] = val;
            r[i] = Convert.ToUInt16(i % num_vertical_scans);
        }

        StartCoroutine(Timer(dt / num_parts));
    }

    IEnumerator Timer(float t)
    {
        yield return new WaitForSeconds(t);
        Scan();
        
        StartCoroutine(Timer(t));
    }

    private void Scan()
    {
        CalculateDirections(part_index * part_size,part_size,rays[part_index]);
        CalculateHits(rays[part_index],hits[part_index]);
        
        part_index += 1;
        if(part_index >= num_parts) {part_index = 0; HandleSensorData(); PublishPointCloud(); }
    }

    [BurstCompile]
    struct CalculateDirectionJob : IJobParallelFor
    {
        public NativeArray<RaycastCommand> rays;
        public int start_index;
        public int num_horizontal_scans;
        public int num_vertical_scans;
        public float horizontal_angle_delta;
        public float vertical_angle_delta;
        public float vertical_fov;
        public Vector3 current_position;
        public Vector3 forward;
        public Vector3 right;
        public Vector3 up;
        
        public void Execute(int index)
        {
            int vertical_dir_index = index % num_vertical_scans;
            int horizontal_dir_index = Mathf.FloorToInt(index / num_vertical_scans) + start_index;

            float horizontal_angle = horizontal_angle_delta * horizontal_dir_index;
            float vertical_angle = vertical_fov / 2f - vertical_dir_index * vertical_angle_delta;

            rays[index] = new RaycastCommand(
                current_position,
                (right * Mathf.Cos(horizontal_angle) + up * Mathf.Sin(vertical_angle) + forward * Mathf.Sin(horizontal_angle)).normalized,
                QueryParameters.Default,
                40f
            );
        }
    }

    private void CalculateDirections(int start_index,int num_scans,NativeArray<RaycastCommand> rays)
    {
        CalculateDirectionJob job = new CalculateDirectionJob(){
            rays = rays,
            start_index = start_index,
            num_horizontal_scans = num_horizontal_scans,
            num_vertical_scans = num_vertical_scans,
            horizontal_angle_delta = 360f / num_horizontal_scans * Mathf.Deg2Rad,
            vertical_angle_delta = vertical_fov / num_vertical_scans * Mathf.Deg2Rad,
            vertical_fov = vertical_fov * Mathf.Deg2Rad,
            current_position = transform.position,
            forward = transform.forward,
            right = transform.right,
            up = transform.up
        };

        JobHandle handle = job.Schedule(num_scans * num_vertical_scans,1);
        handle.Complete();
    }

    private void CalculateHits(NativeArray<RaycastCommand> commands,NativeArray<RaycastHit> hits)
    {
        JobHandle job = RaycastCommand.ScheduleBatch(commands,hits,1);
        job.Complete();
    }

    void HandleSensorData()
    {   
        List<float> x_lst = new List<float>();
        List<float> y_lst = new List<float>();
        List<float> z_lst = new List<float>();
        List<float> i_lst = new List<float>();
        List<ushort> r_lst = new List<ushort>();
        List<float> t_lst = new List<float>();

        for (int i = 0; i < num_parts; i++)
        {
            for (int j = 0; j < part_size * num_vertical_scans; j++)
            {
                if (hits[i][j].collider == null) continue;

                //int index = i * part_size * num_vertical_scans + j;
                Vector3 p = hits[i][j].point - rays[i][j].from;

                Vector3 local = Quaternion.Inverse(transform.rotation) * p;

                x_lst.Add(local.x);
                y_lst.Add(local.y);
                z_lst.Add(local.z);
                i_lst.Add(hits[i][j].collider.GetComponent<Renderer>() != null ? (ushort)Mathf.FloorToInt(hits[i][j].collider.GetComponent<Renderer>().sharedMaterial.color.grayscale*255f) : (ushort)0);
                r_lst.Add(Convert.ToUInt16(i % num_vertical_scans));
                t_lst.Add(i * (1f / freq / num_parts));
                //r[index] = j % num_vertical_scans;
            }
        }

        x = new NativeArray<float>(x_lst.ToArray(),Allocator.TempJob);
        y = new NativeArray<float>(y_lst.ToArray(),Allocator.TempJob);
        z = new NativeArray<float>(z_lst.ToArray(),Allocator.TempJob);
        i = new NativeArray<float>(i_lst.ToArray(),Allocator.TempJob);
        r = new NativeArray<ushort>(r_lst.ToArray(),Allocator.TempJob);
        time = t_lst.ToArray();
        
    }


    private void PublishPointCloud()
    {
        int pointCount = r.Length;

        PointCloud2Msg pointCloudMsg = new PointCloud2Msg
        {
            header = new HeaderMsg
            {
                stamp = new RosMessageTypes.BuiltinInterfaces.TimeMsg(ROSClock._message.clock.sec, ROSClock._message.clock.nanosec),
                frame_id = "velodyne"
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
            Buffer.BlockCopy(BitConverter.GetBytes(this.i[i]), 0, pointCloudMsg.data, i * 22 + 12, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(r[i]), 0, pointCloudMsg.data, i * 22 + 16, 2);
            Buffer.BlockCopy(BitConverter.GetBytes(time[i]), 0, pointCloudMsg.data, i * 22 + 18, 4);
        }

        ros.Publish("/velodyne_points", pointCloudMsg);
    }
}
