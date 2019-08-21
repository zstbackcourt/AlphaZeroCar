using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MLAgents;
using System;
using System.IO;
using EVP;



public class Demo4Agent : Agent
{
    public VehicleController target;
    public GameObject target_object;
    private List<float> agent_state = new List<float>(); // 存储agent state的List
    private Rigidbody agentRb;  // 就是车身，车身是一个刚体
    //private Vector3 relative_pos; // 车与目标点的相对位置
    //private float relative_dis; // 车与目标点的相对距离
    //private float relative_angles; // 车与目标点的相对角度
    private Vector3 car_original_pos=new Vector3(38.45f,0.0190f,10.78f); // 车的初始位置
    private float car_original_eulerAngles_y=10f;// 车的初始欧拉角
    private float init_dis; // 车与目标点的初始距离
    private float pre_dis; // 上一个step时，车与库的距离
    private Camera camera ;
    private RoadRayPerception rayPer;
    private Utils utils;
    private int step_count;// 时间（其实就是用来step计数的)
    //距离信息
    private float complete_dis = 0.55f; // 完成的距离 
    private float previous_dis; // 上一个step时，车离目标点的距离
    //角度
    private float angle_scale = 10f; //入库时车头与目标点的最大角度差
    private float target_object_eulerAngles_y; //目标点的欧拉角_y
    //奖励和惩罚
    private float collider_punish = -4f; // 撞车惩罚
    private float finish_reward = 1.5f;

    private float areaScale = 20f;
    private int count = 0;
    private string[] detectableObjects = { "OBSTACLE", "CAR" };

    private Queue<float> steerInputChange = new Queue<float>(); // 记录轮胎转向变化的队列
    private float previous_steerInput; // 保存上一次的轮胎转向
    
    private float steerInput_obs;//作为obs输入的轮胎转向

    // 用于正向lstm的课程学习阶段目标状态
    private List<float> curr_eulerAngles_y = new List<float>(); 
    private List<Vector3> curr_relative_pos = new List<Vector3>();


    public float RelativeAngles()
        // 计算当前车与目标点的相对角度
    {
        float angle = Mathf.Atan2(agentRb.velocity.x, agentRb.velocity.z); // 车头方向，即车的速度方向
        float angle_target = Mathf.Atan2(target_object.transform.position.x - transform.position.x, target_object.transform.position.z - transform.position.z);
        return angle_target - angle;
    }

    public float RelativeDis()
    {
        return Vector2.Distance(new Vector2(this.transform.position.x, this.transform.position.z), new Vector2(target_object.transform.position.x, target_object.transform.position.z));
    }

    public float RandomFloat(float num)
    {
        return UnityEngine.Random.Range(-num, num);
    }

    public override void InitializeAgent()
    {
        Debug.Log("InitializeAgent");
        base.InitializeAgent();
        camera = Camera.main;
        rayPer = GetComponent<RoadRayPerception>();
        agentRb = GetComponent<Rigidbody>();
        init_dis = Vector2.Distance(new Vector2(car_original_pos.x, car_original_pos.z), new Vector2(target_object.transform.position.x,target_object.transform.position.z));
        pre_dis = init_dis;
        step_count = 0;
        steerInput_obs = 0f;
        target_object_eulerAngles_y = target_object.transform.eulerAngles.y;
        previous_steerInput = 0f;

        curr_relative_pos.Add(new Vector3(-3.9f + RandomFloat(0.1f), 0f, -8.6f + RandomFloat(0.05f)));
        curr_eulerAngles_y.Add(10.3f + RandomFloat(0.15f));

        curr_relative_pos.Add(new Vector3(-3.8f + RandomFloat(0.1f), 0f, -7.5f + RandomFloat(0.05f)));
        curr_eulerAngles_y.Add(13.8f + RandomFloat(0.3f));

        curr_relative_pos.Add(new Vector3(-3.4f + RandomFloat(0.1f), 0f, -5.5f + RandomFloat(0.05f)));
        curr_eulerAngles_y.Add(20.2f + RandomFloat(0.25f));

        curr_relative_pos.Add(new Vector3(-2.5f + RandomFloat(0.1f), 0f, -3.0f + RandomFloat(0.05f)));
        curr_eulerAngles_y.Add(25.5f + RandomFloat(0.3f));

        curr_relative_pos.Add(new Vector3(-1.5f + RandomFloat(0.1f), 0f, -0.9f + RandomFloat(0.1f)));
        curr_eulerAngles_y.Add(18.4f + RandomFloat(0.15f));

    }

    public override void CollectObservations()
    {
        // 状态的前两维是车的坐标x,z，不参与policy只是用来recover状态的
        agent_state.Clear(); // 清除上一帧的环境信息
        //添加相对角度,车的当前速度，轮胎转向，同目标点的相对距离
        float relative_angles = RelativeAngles();
        Vector3 relative_pos = target_object.transform.position - this.transform.position;
        float relative_dis = RelativeDis();
        agent_state.Add(this.transform.position.x);
        agent_state.Add(this.transform.position.z);
        for (int i = 0; i < 5; i++)
        {
            agent_state.Add(agentRb.velocity.x);
            agent_state.Add(agentRb.velocity.z);
            agent_state.Add(relative_angles);
            agent_state.Add(steerInput_obs);
            agent_state.Add(relative_dis);
        }
      
        //添加周围信息
        Add_raycast(ref agent_state);
        AddVectorObs(agent_state);

    }


    // agent执行的动作
    // <param name="vectorAction[0]">转向，-1到1
    // <param name="vectorAction[1]">油门/刹车，正为油门负为刹车
    // <param name="vectorAction[2]">用来判断是恢复状态还是执行动作
    // <param name="vectorAction[3,4]">速度x,z
    // <param name="vectorAction[5,6]">坐标x,z
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (vectorAction[2] == 1) // 表示用来恢复状态而不是执行动作
        {
            Debug.Log("recover state!");
            // 恢复坐标
            this.transform.position = new Vector3(vectorAction[3], 0.0f, vectorAction[4]);
            // 恢复速度
            this.agentRb.velocity = new Vector3(vectorAction[5], 0.0f, vectorAction[6]);

        }
        else if (vectorAction[2] == 0)
        {
            // 获得当前速度
            float currentVelocity = (float)System.Math.Sqrt((double)(this.agentRb.velocity.x) * (this.agentRb.velocity.x) + (double)(this.agentRb.velocity.z) * (this.agentRb.velocity.z));
            // 用于判断速度方向
            Vector3 localVel = transform.InverseTransformDirection(this.agentRb.velocity);

            // 车的当前欧拉角
            float car_eulerAngles_y = this.transform.eulerAngles.y;
            // 当前离目标点的距离
            float current_dis = RelativeDis();
            // 车当前的相对位置
            Vector3 relative_pos = target_object.transform.position - this.transform.position;
            //如果超过最长步长还没有入库就直接done
            if (step_count > 512)
            {
                if (init_dis <= current_dis)
                {
                    AddReward(-2f);
                }
                if (current_dis > complete_dis)
                {
                    AddReward(-1f);
                }
                float reward = GetCumulativeReward();
                Debug.Log("(最长时间步)当前总的reward:" + reward);
                Done();
            }
            step_count++;
            //if (current_dis < pre_dis)
            //    // 如果当前距离大于上一个step的距离就给与一个奖励
            //{
            //    (pre_dis - current_dis) / init_dis;
            //}
            AddReward((pre_dis - current_dis) / init_dis);
            pre_dis = current_dis;
            if (car_eulerAngles_y > 45)
            // 倒车不可能欧拉角超过45度，所以只要超过就施加惩罚
            {
                AddReward(-(car_eulerAngles_y / 180f) * 0.1f);
            }
            //Debug.Log((pre_dis - current_dis) / init_dis);
            if (step_count % 4 == 0)
            {
                target.steerInput = vectorAction[0];
                steerInput_obs = target.steerInput;
                // 判断本次打方向和上次打方向的幅度变化
                float steerChange = (System.Math.Abs(vectorAction[0] - previous_steerInput)) * 45f;
                if (steerChange > 15)
                // 本次轮胎方向和上一次轮胎方向相差15度以上
                {
                    AddReward(-0.01f * (float)System.Math.Floor(steerChange / 15f));
                }
                if (steerInputChange.Count < 3)
                {
                    steerInputChange.Enqueue(vectorAction[0]);
                }
                else
                {

                    float steer_1 = steerInputChange.Dequeue();

                    float steer_2 = steerInputChange.Dequeue();

                    float steer_3 = steerInputChange.Dequeue();

                    if (steer_1 * steer_2 < 0 && steer_2 * steer_3 < 0)
                    {
                        AddReward(-0.01f);
                    }
                    steerInputChange.Enqueue(steer_2);
                    steerInputChange.Enqueue(steer_3);
                    steerInputChange.Enqueue(vectorAction[0]);
                }
                previous_steerInput = target.steerInput;

            }
            else
            {
                target.steerInput = previous_steerInput;
                steerInput_obs = target.steerInput;
            }
            // 判断此时应该是油门还是刹车
            if (vectorAction[1] > 0f)
            {
                // 此时是油门
                // 判断此时是向前还是向后
                if (localVel.z > 0f)
                {
                    // 速度向前
                    target.throttleInput = vectorAction[1];
                    target.brakeInput = 0f;

                }
                else
                {
                    // 速度向后
                    target.throttleInput = -1.0f * vectorAction[1];
                    target.brakeInput = 0f;
                }

            }
            else if (vectorAction[1] == 0f)
            {
                // 此时油门刹车均为0
                target.brakeInput = 0f;
                target.throttleInput = 0f;
            }
            else if (vectorAction[1] < 0f)
            {
                // 此时是刹车
                target.throttleInput = 0f;
                target.brakeInput = System.Math.Abs(vectorAction[1]);//刹车是控制在0-1
            }
            // 如果到达目标点
            if (current_dis < complete_dis)
            {
                AddReward(1.5f);
                float relative_eulerAngles_y = car_eulerAngles_y - target_object_eulerAngles_y;
                AddReward(Mathf.Abs(relative_eulerAngles_y - 180.0f) / 180.0f - 1.0f);//角度差带来的惩罚
                Debug.Log(Mathf.Abs(relative_eulerAngles_y - 180.0f) / 180.0f - 1.0f);
                AddReward(-currentVelocity / 10.0f);//速度过大带来的惩罚
                AddReward(-current_dis / 10.0f);//与目标点的距离带来的惩罚
                float reward = GetCumulativeReward();
                Debug.Log("完成入库，当前总的reward:" + reward);
                Done(); // 完成
            }

        }
    }


    public override void AgentReset()
    {

        //Debug.Log("AgentResst");
        steerInputChange.Clear();
        this.transform.position = car_original_pos;
        this.transform.eulerAngles = new Vector3(0, car_original_eulerAngles_y, 0);
        this.agentRb.angularVelocity = Vector3.zero;
        this.agentRb.velocity = Vector3.zero;
        this.step_count = 0;
        this.previous_steerInput = 0f;
        this.pre_dis = init_dis;
        //Debug.Log("init_dis:" + init_dis);
        //Debug.Log("pre_dis:" + pre_dis);

    }

    //发生碰撞时
    public void OnCollisionEnter(Collision collision)
    {
        for (int i = 0; i < detectableObjects.Length; i++)
        {
            if (collision.gameObject.CompareTag(detectableObjects[i]))
            {
                AddReward(collider_punish);
                float reward = GetCumulativeReward();
                Debug.Log("(撞车)当前总的reward" + reward);
            }
            Done();
        }
    }

   

    public void Add_raycast(ref List<float> x)
    {
        //计算360度信息
        const float rayDistance = 10f;
        //360度信息，但是会卡
        float[] rayAngles = new float[72];
        int j = 0;
        for (float i = -179f; i < 180; i += 5f)
        {
            rayAngles[j] = i;
            j++;
        }

        //接受返回的信息 一个向量(dis1+type6+angle1); 即(1+(5(以上五种)+1(nothing))+1)*72=8*72
        List<float> perceptions_info = new List<float>();
        perceptions_info = rayPer.Perceive(rayDistance, rayAngles, detectableObjects, 0.5f, 0.5f);

        foreach (float i in perceptions_info)
        {
            x.Add(i);
        }
    }

}
