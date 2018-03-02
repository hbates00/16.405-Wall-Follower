#!/usr/bin/env python2

import rospy
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32


class WallFollower:
    # Imports ROS parameters from the "params.yaml" file.

    #  ---NODE INFORMATION---  #
    _NODE_NAME =            "WallFollower"
    _LOG_LEVEL =            rospy.DEBUG
    _QUEUE_SIZE =           10

    #  ---SCAN INFORMATION---  #
    _SCAN_TOPIC =           rospy.get_param("/wall_follower/scan_topic")
    _SCAN_TYPE =            LaserScan
    _LOOK_ANGLE_BACK =      np.pi/9 #[RADIANS]
    _LOOK_ANGLE_FORWARD =   88*np.pi/180

    #  ---DRIVE INFORMATION---  #
    _DRIVE_TOPIC =          rospy.get_param("wall_follower/drive_topic")
    _DRIVE_TYPE =           AckermannDriveStamped

    #  ---SEGMENT VISUALIZATION INFORMATION---  #
    _MARKER_TOPIC =         "markers"
    _MARKER_TOPIC_TYPE =    Marker

    #  ---STATE INFORMATION---  #
    _SIDE =                 rospy.get_param("wall_follower/side") #  +1 = LEFT SIDE, -1 = RIGHT SIDE
    _VELOCITY =             rospy.get_param("wall_follower/velocity", 500)
    _DESIRED_DISTANCE =     rospy.get_param("wall_follower/desired_distance")
    _SEGMENTS_MAX_RATE_HZ = 10


    #  ---GAINS---  #
    #_LAMBDA = 1.0
    _Kp = 1.5
    _Kd = 25
    _K_theta = 2.5
    #_Ki = 0.05

    def __init__(self):

        #  Initializes node
        rospy.init_node(self._NODE_NAME, log_level = self._LOG_LEVEL)

        #  Defines the rate at which the node runs
        rate = rospy.Rate(self._SEGMENTS_MAX_RATE_HZ)

        #  Initializes subscriber and publisher
        rospy.Subscriber(self._SCAN_TOPIC, self._SCAN_TYPE, self.callback)

        self.error = 0
        self.error_diff = 0
        self.error_int = 0

        #  Initializes drive state publisher
        self.pub = rospy.Publisher(self._DRIVE_TOPIC, 
                                   self._DRIVE_TYPE, 
                                   queue_size = self._QUEUE_SIZE)

        #  Initializes visualization publisher
        self.pub_marker = rospy.Publisher(self._MARKER_TOPIC, 
                                          self._MARKER_TOPIC_TYPE, 
                                          queue_size = self._QUEUE_SIZE)

    def callback(self, state):
     
        #  Checking to make sure list is even, forcing if not
        # if len(xyz_tup) % 2 != 0:
        #     xyz_tup = xyz_tup[:-1]

        dist, theta, x_fit, y_fit, d_vec = self.get_controller_values(state)

        self.e_diff = (dist - self._DESIRED_DISTANCE) - self.error
        #self.error_int += self.error
        self.error = dist - self._DESIRED_DISTANCE

        delta = self._SIDE * (self._Kp * self.error + self._Kd * self.error_diff) + self._K_theta * (theta - (np.pi * self._SIDE)/2)

        
        segments = [(x_fit[0], y_fit[0], 0), (x_fit[-1], y_fit[-1], 0)]

        marker = Marker()
        marker.header.frame_id = "laser"
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.1
        marker.id = 0
        marker.color.r = 0
        marker.color.g = 0.5
        marker.color.b = 0
        marker.color.a = 1
        marker.points = [Point(x, y, z) for (x, y, z) in segments]

        self.pub_marker.publish(marker)
        
        #  Initializes the message to be published
        drive_state = AckermannDriveStamped()

        #  Sets drive states
        drive_state.drive.steering_angle = delta
        drive_state.drive.steering_angle_velocity = 0
        drive_state.drive.speed = 2*self._VELOCITY
        drive_state.drive.acceleration = 0
        drive_state.drive.jerk = 0

        #  Publishes drive states
        self.pub.publish(drive_state)

    def get_controller_values(self, state):

        #  Finds the radius and angle values of the entire sweep
        full_rtheta = self.get_rtheta(state)

        #  Finds slice of rtheta we want
        slice_rtheta = self.slice_rtheta(state, full_rtheta)

        #  Cleans it and removes infinite data points
        filtered_rtheta = self.filter_infinite(slice_rtheta, state.range_max, 0.1)

        #  Gets cartesian coordinates from cleaned rtheta data
        xyz_tup = self.get_cartesian(filtered_rtheta)

        x_fit, y_fit = self.line_fit(xyz_tup)

        #  Finds the perpendicular distance vector
        d_vec = self.find_distance(x_fit, y_fit)

        #  Finds the perpendicular distance
        dist = np.sqrt(d_vec[0]**2 + d_vec[1]**2)

        #  Finds the angle of the perpendicular distance
        theta = self.find_direction(d_vec)

        return(dist, theta, x_fit, y_fit, d_vec)
    
    def get_rtheta(self, state):

        #  Gets range of indexes from min angle to max angle
        indexes = np.array(list(range(0, len(state.ranges) - 1)))
        
        #  Finds angle for each range
        theta = state.angle_increment * indexes + state.angle_min

        #  Returns as a list of tuples (range, angle)
        return np.array(zip(state.ranges, theta))

    def slice_rtheta(self, state, rtheta):

        #  Finds number of indexes within the look angle
        num_look_index_forward = int(self._LOOK_ANGLE_FORWARD/state.angle_increment)
        num_look_index_back = int(self._LOOK_ANGLE_BACK/state.angle_increment)

        #  Number of indexes on the RIGHT side
        num_index_right = np.absolute((int(state.angle_min/state.angle_increment)))

        #  RIGHT SIDE
        if self._SIDE == -1:

            #  Number of indexes to the y axis
            num_index_y = int((np.abs(state.angle_min) - np.pi/2)/state.angle_increment)

            #  Finds minimum and maximum look indixes
            min_search_index = int(num_index_y - num_look_index_back)
            max_search_index = int(num_index_y + num_look_index_forward)

            return rtheta[min_search_index:max_search_index]

        #  LEFT SIDE
        else:

            #  Number of indexes to the y axis
            num_index_y = int((((state.angle_max) - np.pi/2)/state.angle_increment) + np.pi/state.angle_increment)

            min_search_index = int(num_index_y - num_look_index_forward)
            max_search_index = int(num_index_y + num_look_index_back)

            return rtheta[min_search_index:max_search_index]

    def filter_infinite(self, rtheta, max_range, epsilon):

        #  Finds only the range values within max_range
        result = [(r, theta) for (r, theta) in rtheta if r < (max_range - epsilon)]

        return result

    def get_cartesian(self, rtheta):

        #  Converts rtheta to cartesian coordinates
        xyz_tup = [(r * np.cos(theta), r * np.sin(theta), 0) for (r, theta) in rtheta]

        return xyz_tup

    def line_fit(self, xyz_tup):

        x, y, z = zip(*xyz_tup)

        #  Function to which we fit data in slice
        def func(x, a, b):

            #  A line
            return(a*x + b)

        #  Finds best fit line of data
        popt, pcov = curve_fit(func, x, y)

        #  Finds smooth, fitted y values
        smooth = func(np.array(x), *popt)

        #  Returns x coordinates and smoothed y coordinates
        return(x, smooth)

    def find_distance(self, x, y):

        #  RIGHT SIDE
        if self._SIDE == -1:

            #  Vector from car to end of line
            a = np.array([x[0], y[0]])
            
            #  Vector of line
            b = np.array([(x[0] - x[-1]), (y[0] - y[-1])])

            #  Projection of a onto b
            c = (a[0] * b[0] + a[1] * b[1]) / (np.sqrt(b[0]**2 + b[1]**2)) * b / (np.sqrt(b[0]**2 + b[1]**2))

            #  Perpendicular vector from car to line
            d = a - c

        #  LEFT SIDE
        else: 

            #  Vector from car to end of line
            a = np.array([x[-1], y[-1]])
            
            #  Vector of line
            b = np.array([(x[-1] - x[0]), (y[-1] - y[0])])

            #  Projection of a onto b
            c = (a[0] * b[0] + a[1] * b[1]) / (np.sqrt(b[0]**2 + b[1]**2)) * b / (np.sqrt(b[0]**2 + b[1]**2))

            #  Perpendicular vector from car to line
            d = a - c


        return(d)


    def find_direction(self, d):

        #  Finds the angle of the wall to the car
        return(np.arctan2(d[1], d[0]))


if __name__ == "__main__":
    wall_follower = WallFollower()
    rospy.spin()
