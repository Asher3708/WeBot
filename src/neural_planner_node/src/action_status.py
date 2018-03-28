class ActionStatus(object):
    """This class is used to manage action."""
    def __init__(self):
        self.stop = 0
        self.forward = 1
        self.left_front = 2
        self.right_front = 3
        self.left_back = 4
        self.right_back = 5

    def vel_to_action(self, vel_forward, vel_rotate):
        """According forward velocity and velocity of rotation, speculate demonstration action.
        Args:
            vel_forward:
            vel_rotatie:
        Return:
            return a action number.
        """
        threshold_forward = 0.1
        threshold_roate = 0.4
        action = -1
        if abs(vel_forward) < threshold_forward:
            if abs(vel_rotate) < threshold_roate: # stop
                action = self.stop
            elif vel_rotate < 0: # move to right back
                action = self.right_back
            else: # move to left back
                action = self.left_back
        else:
            if abs(vel_rotate) < threshold_roate: # forward
                action = self.forward
            elif vel_rotate < 0: # move to left front
                action = self.left_front
            else: # move to right front
                action = self.right_front

        return action

    def action_to_angle(self, action):
        """Transform Action to a angle."""
        if action == self.stop:
            angle = 0
            print "WARN: action-status is stop."
        elif action == self.forward:
            angle = 0
        elif action == self.left_front:
            angle = -45+360
        elif action == self.right_front:
            angle = 45
        elif action == self.left_back:
            angle = -135+360
        elif action == self.right_back:
            angle = 135
        else:
            raise Exception("InvalidActionStatus","This should never be arrived.")

        return angle
