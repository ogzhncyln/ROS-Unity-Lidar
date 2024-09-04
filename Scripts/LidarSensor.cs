using System.Collections;
using UnityEngine;
using Unity.Mathematics;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using System;
using Unity.Collections;
using Unity.Jobs;
using System.Linq;

public class LidarSensor : MonoBehaviour
{
    public float freq = 10f;
    public float range = 40f;
    public int num_horizontal_scans = 1800, num_vertical_scans = 16;
    public float horizontal_delta_ang = 360f / 1800f, vertical_delta_ang = 2f;

    public int num_parts = 6;

    public ComputeShader computeShader;
    private ComputeBuffer directionsBuffer;

    private NativeArray<RaycastCommand> ray_cmds;
    private NativeArray<RaycastHit> results;

    private float[] arr_x, arr_y, arr_z;
    private int[] arr_i;

    private ROSConnection ros;

    private void Scan(int start_indx, int num_scans)
    {
        int num_rays = num_horizontal_scans * num_vertical_scans;
        Vector3[] directions = new Vector3[num_rays];
        int kernelHandle = computeShader.FindKernel("CSMain");

        computeShader.SetInt("num_h_scans", num_horizontal_scans);
        computeShader.SetInt("num_v_scans", num_vertical_scans);
        computeShader.SetInt("start_indx", start_indx);
        computeShader.SetFloat("ang_h_delta", horizontal_delta_ang * Mathf.Deg2Rad);
        computeShader.SetFloat("ang_v_delta", vertical_delta_ang * Mathf.Deg2Rad);
        computeShader.SetFloat("ang_v_max", (num_vertical_scans - 1) * vertical_delta_ang * Mathf.Deg2Rad);

        directionsBuffer.SetData(directions);
        computeShader.SetBuffer(kernelHandle, "directions", directionsBuffer);
        computeShader.Dispatch(kernelHandle, Mathf.CeilToInt(num_scans / 10.0f), Mathf.CeilToInt(num_vertical_scans / 8.0f), 1);

        directionsBuffer.GetData(directions);

        for (int i = 0; i < num_rays; i++)
        {
            ray_cmds[i] = new RaycastCommand(
                transform.position,
                (transform.forward * directions[i].z + transform.right * directions[i].x + transform.up * directions[i].y).normalized,
                QueryParameters.Default,
                range
            );
        }

        JobHandle handle = RaycastCommand.ScheduleBatch(ray_cmds, results, 1);
        handle.Complete();

        for (int i = 0; i < num_rays; i++)
        {
            RaycastHit hit = results[i];
            if (hit.collider != null)
            {
                arr_x[i] = hit.point.x;
                arr_y[i] = hit.point.y;
                arr_z[i] = hit.point.z;
                arr_i[i] = 255;
            }
        }
        
    }

    public void PublishPointCloud()
    {
        int pointCount = num_horizontal_scans * num_vertical_scans;

        PointCloud2Msg pointCloudMsg = new PointCloud2Msg
        {
            header = new HeaderMsg
            {
                stamp = new RosMessageTypes.BuiltinInterfaces.TimeMsg(0, 0),
                frame_id = "velodyne"
            },
            height = 1,
            width = (uint)pointCount,
            fields = new[]
            {
                new PointFieldMsg { name = "x", offset = 0, datatype = PointFieldMsg.FLOAT32, count = 1 },
                new PointFieldMsg { name = "y", offset = 4, datatype = PointFieldMsg.FLOAT32, count = 1 },
                new PointFieldMsg { name = "z", offset = 8, datatype = PointFieldMsg.FLOAT32, count = 1 },
                new PointFieldMsg { name = "intensity", offset = 12, datatype = PointFieldMsg.INT8, count = 1 }
            },
            is_bigendian = false,
            point_step = 13,
            row_step = (uint)(13 * pointCount),
            is_dense = true,
            data = new byte[13 * pointCount]
        };

        for (int i = 0; i < pointCount; i++)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(arr_x[i]), 0, pointCloudMsg.data, i * 13, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(arr_z[i]), 0, pointCloudMsg.data, i * 13 + 4, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(arr_y[i]), 0, pointCloudMsg.data, i * 13 + 8, 4);
            pointCloudMsg.data[i * 13 + 12] = (byte)arr_i[i];
        }

        ros.Publish("/velodyne_points", pointCloudMsg);
    }

    private int part_indx;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<PointCloud2Msg>("/velodyne_points", 1);

        arr_x = new float[num_vertical_scans * num_horizontal_scans];
        arr_y = new float[num_vertical_scans * num_horizontal_scans];
        arr_z = new float[num_vertical_scans * num_horizontal_scans];
        arr_i = new int[num_vertical_scans * num_horizontal_scans];

        ray_cmds = new NativeArray<RaycastCommand>(num_vertical_scans * num_horizontal_scans, Allocator.Persistent);
        results = new NativeArray<RaycastHit>(num_vertical_scans * num_horizontal_scans, Allocator.Persistent);

        directionsBuffer = new ComputeBuffer(num_horizontal_scans * num_vertical_scans, sizeof(float) * 3);

        float dt = 1f / freq / num_parts;
        StartCoroutine(Timer(dt));
    }

    private void OnDestroy()
    {
        directionsBuffer.Release();
        ray_cmds.Dispose();
        results.Dispose();
    }

    IEnumerator Timer(float p)
    {
        yield return new WaitForSeconds(p);
        Scan((num_horizontal_scans / num_parts) * part_indx, num_horizontal_scans / num_parts);
        part_indx += 1;

        if (part_indx >= num_parts)
        {
            part_indx = 0;
            PublishPointCloud();
            arr_x = new float[num_vertical_scans * num_horizontal_scans];
            arr_y = new float[num_vertical_scans * num_horizontal_scans];
            arr_z = new float[num_vertical_scans * num_horizontal_scans];
            arr_i = new int[num_vertical_scans * num_horizontal_scans];
        }

        StartCoroutine(Timer(p));
    }
}

