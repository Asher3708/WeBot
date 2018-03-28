#include <pluginlib/class_list_macros.h> // For PLUGINLIB_EXPORT_CLASS
#include <global_action_planner/grid_action_planner.h>
#include <math.h>
#include <neural_planner_node/ActionOrientation.hpp>

/* Register this planner as a BaseGlobalPlanner plugin. */ 
PLUGINLIB_EXPORT_CLASS(grid_action_planner::GridActionPlanner, nav_core::BaseGlobalPlanner)

namespace grid_action_planner
{
    GridActionPlanner::GridActionPlanner():is_initialized_(false){
    }

    void GridActionPlanner::initialize(std::string name, costmap_2d::Costmap2DROS *costmap_ros){
        stride_ = kStride;
        ros::NodeHandle n;
        client_ = n.serviceClient<neural_planner_node::...>("");
        is_initialized_ = true;
    }

    float distance_square(const geometry_msgs::PoseStamped &start
                        , const geometry_msgs::PoseStamped &end)
    {
        double delta_x = end.pose.position.x - start.pose.position.x;
        double delta_y = end.pose.position.y - end.pose.position.y;
        return delta_x*delta_x + delta_y*delta_y;
    }

    bool GridActionPlanner::makePlan(const geometry_msgs::PoseStamped &_start,
                                     const geometry_msgs::PoseStamped &_goal,
                                     std::vector<geometry_msgs::PoseStamped> &plan)
    {
        if(!is_initialized_){
            std::cerr<<"Error: Mistake to use a uninitialized GridActionPlanner object."<<std::endl;
            return false;
        }
        // If it is close enough between start and goal, it is unnecessary to route planning.
        if(...){ // 是否在终点附近
            // 在终点附近时，则根据一些工程学方法完成导航。例如：引导信号
        }
        else{ // 不在终点附近时，依靠global_path_planner
            // Request service of path planning
            req = ...
            action_angle = 0;
            if(client.call(srv)){
                action_angle = srv.response.ret;
            }else{
                ROS_ERROR("Failed");
                // 失败处理
            }
            // Transform action angle to path
            //...转换可以参考V7.0
        }
        // Fill goal into path.
        plan.push_back(goal);
        return true;
    }

}