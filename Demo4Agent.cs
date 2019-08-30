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
    // 基本设置
    private float areaScale = 20f;
    public GameObject target_object; // 目标点
    private List<float> agent_state = new List<float>(); // 存储obs的列表
    private Rigidbody agentRb; // 车
    public VehicleController target; // 车
    private Camera camera;
    private RoadRayPerception rayPer;
    private Utils utils;

    private Vector3 car_original_pos = new Vector3(38.45f, 0.0190f, 10.78f); // 车的初始位置
    private float car_original_eulerAngles_y = 0.1f;// 车的初始欧拉角
    private float target_object_eulerAngles_y; //目标点的欧拉角_y
    private float init_dis; // 车与目标点的初始距离
    private float pre_dis; // 上一个step时，车与库的距离 
    private float cur_dis; // 当前相对距离
    private int step_count;// 时间（其实就是用来step计数的)
    private float complete_dis = 0.55f; // 完成的距离 
    private float complete_angle_scale = 10f; //入库时车头与目标点的最大角度差
    private float collider_punish = -4f; // 撞车惩罚
    private float finish_reward = 1.5f; // 完成奖励
    private string[] detectableObjects = { "OBSTACLE", "CAR" };
    private float pre_steerInput; // 上一个step的轮胎转向
    private float cur_steerInput;

    public float RelativeAngles()
    // 计算当前车与目标点的相对角度
    {
        float angle = Mathf.Atan2(agentRb.velocity.x, agentRb.velocity.z); // 车头方向，即车的速度方向
        float angle_target = Mathf.Atan2(target_object.transform.position.x - transform.position.x, target_object.transform.position.z - transform.position.z);
        return angle_target - angle;
    }

    public float RelativeDis()
    // 计算当前车与目标点的相对距离
    {
        return Vector2.Distance(new Vector2(this.transform.position.x, this.transform.position.z), new Vector2(target_object.transform.position.x, target_object.transform.position.z));
    }

    public float RandomFloat(float num)
    // 随机一个随机数
    {
        return UnityEngine.Random.Range(-num, num);
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

    public override void InitializeAgent()
    {
        // 
        Debug.Log("InitializeAgent");
        base.InitializeAgent();
        camera = Camera.main;
        rayPer = GetComponent<RoadRayPerception>();
        agentRb = GetComponent<Rigidbody>();
        init_dis = Vector2.Distance(new Vector2(car_original_pos.x, car_original_pos.z), new Vector2(target_object.transform.position.x, target_object.transform.position.z));
        pre_dis = init_dis;
        cur_dis = init_dis;
        step_count = 0;
        target_object_eulerAngles_y = target_object.transform.eulerAngles.y; // 目标点的欧拉角(即车的欧拉角要和该欧拉角相差不超过10度)
        pre_steerInput = 0f;
        cur_steerInput = 0f;
    }

    public override void CollectObservations()
    {
        
        agent_state.Clear(); // 清除上一帧的环境信息
        float relative_angles = RelativeAngles(); // 当前相对角度
        cur_dis = RelativeDis(); // 当前的相对距离

        agent_state.Add(step_count);
        agent_state.Add(pre_dis); // 添加前一个step的相对距离
        agent_state.Add(pre_steerInput); //添加前一个step的steerInput
        agent_state.Add(this.transform.eulerAngles.y); // 当前车的欧拉角
        agent_state.Add(this.transform.position.x); // 当前的车的坐标
        agent_state.Add(this.transform.position.z);
        agent_state.Add(agentRb.velocity.y);
        for (int i = 0; i < 5; i++)
        {
            agent_state.Add(agentRb.velocity.x); // 当前的车的速度
            agent_state.Add(agentRb.velocity.z);
            agent_state.Add(cur_steerInput); // 当前的steerInput
            agent_state.Add(cur_dis);
            agent_state.Add(relative_angles);
        }
        // Debug.Log("*cur_steer" + cur_steerInput + " " + "pre_steer" + pre_steerInput);
        //添加周围信息
        Add_raycast(ref agent_state);
        AddVectorObs(agent_state);
        pre_steerInput = cur_steerInput;
        //Debug.Log("**cur_steer" + cur_steerInput + " " + "pre_steer" + pre_steerInput);

    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (vectorAction[2] == 1)
            // 恢复状态
        {
            step_count = (int)(vectorAction[3]);
            pre_dis = vectorAction[4];
            pre_steerInput = vectorAction[5];
            this.transform.eulerAngles = new Vector3(0.0f, vectorAction[6], 0.0f);
            this.transform.position = new Vector3(vectorAction[7], 0.0f, vectorAction[8]);
            this.agentRb.velocity = new Vector3(vectorAction[10], vectorAction[9], vectorAction[11]);
            cur_steerInput = vectorAction[12];
            cur_dis = vectorAction[13];
        }
        else
        {

            // 获得当前速度
            float currentVelocity = (float)System.Math.Sqrt((double)(this.agentRb.velocity.x) * (this.agentRb.velocity.x) + (double)(this.agentRb.velocity.z) * (this.agentRb.velocity.z));
            // 用于判断速度方向
            Vector3 localVel = transform.InverseTransformDirection(this.agentRb.velocity);
            // 车的当前欧拉角
            float car_eulerAngles_y = this.transform.eulerAngles.y;
            // Debug.Log(this.transform.eulerAngles.x + "," + this.transform.eulerAngles.y + "," + this.transform.eulerAngles.z);
            // 当前离目标点的距离
            cur_dis = RelativeDis();
            // 车当前的相对位置
            Vector3 relative_pos = target_object.transform.position - this.transform.position;
            if (step_count > 512)
            {
                if (init_dis <= cur_dis)
                {
                    AddReward(-2f);
                }
                if (cur_dis > complete_dis)
                {
                    AddReward(-1f);
                }
                float reward = GetCumulativeReward();
                Debug.Log("(最长时间步)当前总的reward:" + reward);
                Done();
            }
            else
            {
                step_count++;
               //Debug.Log("执行前："+this.transform.position);
                AddReward((pre_dis - cur_dis) / init_dis);
                pre_dis = cur_dis;
                if (car_eulerAngles_y > 45)
                // 倒车不可能欧拉角超过45度，所以只要超过就施加惩罚
                {
                    AddReward(-(car_eulerAngles_y / 180f) * 0.1f);
                }
                target.steerInput = vectorAction[0];
                cur_steerInput = target.steerInput;
                // Debug.Log("cur_steer" + cur_steerInput + " " + "pre_steer" + pre_steerInput);
                // 判断本次打方向和上次打方向的幅度变化
                float steerChange = (System.Math.Abs(cur_steerInput - pre_steerInput)) * 45f;
                if (steerChange > 15)
                // 本次轮胎方向和上一次轮胎方向相差15度以上
                {
                    AddReward(-0.01f * (float)System.Math.Floor(steerChange / 15f));
                }
                // pre_steerInput = cur_steerInput;
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
                if (cur_dis < complete_dis)
                {
                    AddReward(1.5f);
                    float relative_eulerAngles_y = car_eulerAngles_y - target_object_eulerAngles_y;
                    AddReward(Mathf.Abs(relative_eulerAngles_y - 180.0f) / 180.0f - 1.0f);//角度差带来的惩罚
                    Debug.Log(Mathf.Abs(relative_eulerAngles_y - 180.0f) / 180.0f - 1.0f);
                    AddReward(-currentVelocity / 10.0f);//速度过大带来的惩罚
                    AddReward(-cur_dis / 10.0f);//与目标点的距离带来的惩罚
                    float reward = GetCumulativeReward();
                    Debug.Log("完成入库，当前总的reward:" + reward);
                    Done(); // 完成
                }
                //Debug.Log("执行后:" + this.transform.position);
            }
        }
    }

    public override void AgentReset()
    {

        //Debug.Log("AgentResst");
        this.transform.position = car_original_pos;
        this.transform.eulerAngles = new Vector3(0, car_original_eulerAngles_y, 0);
        this.agentRb.angularVelocity = Vector3.zero;
        this.agentRb.velocity = Vector3.zero;
        step_count = 0;
        pre_steerInput = 0f;
        pre_dis = init_dis;
        cur_dis = init_dis;
        cur_steerInput = 0f;
        //Debug.Log("init_dis:" + init_dis);
        //Debug.Log("pre_dis:" + pre_dis);

    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag(detectableObjects[0]) || collision.gameObject.CompareTag(detectableObjects[1]))
        {
            AddReward(collider_punish);
            float reward = GetCumulativeReward();
            Debug.Log("(撞车)当前总的reward" + reward);
            Done();

        }
        // Debug.Log("Done() 之后step:" + stepCount);
    }

}