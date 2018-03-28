#ifndef GRID_ACTION_PLANNER_H
#define GRID_ACTION_PLANNER_H

#include <ros/ros.h>
#include <costmap_2d/costmap_2d_ros.h>    // For costmap_2d::Costmap2DROS
#include <nav_core/base_global_planner.h> // For nav_core::BaseGlobalPlanner
#include <geometry_msgs/PoseStamped.h>    // For geometry_msgs::PoseStamped
#include <vector> // For std::vector
#include <string> // For std::string

namespace grid_action_planner{

  class GridActionPlanner:public nav_core::BaseGlobalPlanner{
    public:
      GridActionPlanner();

      void initialize(std::string name, costmap_2d::Costmap2DROS *costmap_ros);
      /**
      * @brief  global path planner.
      * @param start: this is useless, we remain it because of the interface of move_base.
      *               we get robot pose information by inner API, rather than formal parameter.
      * @param goal : this is useless for the tiime being. Now, our navigation goal is unique.
      * @param plan : path.
      * @return True if the recovery behaviors were loaded successfully, false otherwise
      */
      bool makePlan(const geometry_msgs::PoseStamped &start,
                      const geometry_msgs::PoseStamped &goal,
                      std::vector<geometry_msgs::PoseStamped> &plan);
    private:
      bool is_initialized_;
      // Action distance along action angle at every time.
      float stride_;
      ros::ServiceClient client_;
  };
};
#endif