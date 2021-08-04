#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import rclpy
from rclpy.node import Node

# from sensor_msgs.msg import Image
from perception_messages.msg import Detections3DArray

# from perception_messages.msg import Detection2D

import time

# import logging
import os
import numpy as np
import sys

sys.path.insert(0, os.getcwd())
from AB3DMOT_libs.model import AB3DMOT


class Sort_Tracking(Node):
    def __init__(self):
        super().__init__("tracking_sort_node")

        # Declare parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("topic_in", "image"),
                ("topic_out", "detections"),
                ("queue_pub", 10),
                ("queue_sub", 10),
            ],
        )

        # get parameters
        topic_in = self.get_parameter("topic_in").get_parameter_value().string_value
        topic_out = self.get_parameter("topic_out").get_parameter_value().string_value
        queue_pub = self.get_parameter("queue_pub").get_parameter_value().integer_value
        queue_sub = self.get_parameter("queue_sub").get_parameter_value().integer_value

        # create subscriber
        self.subscription = self.create_subscription(
            Detections3DArray, topic_in, self.listener_callback, queue_sub
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info("Listening to %s topic" % topic_in)

        # create publisher
        self.publisher_ = self.create_publisher(Detections3DArray, topic_out, queue_pub)

        # initialize variables
        self.last_time = time.time()
        self.last_update = self.last_time

        # Detector initialization
        self.model = self._init_model()
        self.get_logger().info("tracking model initialized")

    def listener_callback(self, msg):
        # transform message
        detections = self._msg2detections(msg)

        if len(detections) > 0:
            # segment image
            input_tracking = {
                "dets": detections,
                "info": np.array([None for _ in range(len(detections))])[..., np.newaxis],
            }
            track_bbs_ids, matched_ids = self.model.update(input_tracking)

            # publish results
            # if len(track_bbs_ids) > 0 and len(track_bbs_ids) == len(detections):
            if len(track_bbs_ids) > 0:
                self.get_logger().info(
                    f"len bbox {len(track_bbs_ids)}, len detection {len(detections)}"
                )
                assert len(track_bbs_ids) == len(detections)
                tracked_msg = self._create_bbox_message(msg, track_bbs_ids, matched_ids)
            else:
                tracked_msg = Detections3DArray()
        else:
            tracked_msg = Detections3DArray()

        tracked_msg.header = msg.header
        self.publisher_.publish(tracked_msg)

        # compute true fps
        curr_time = time.time()
        fps = 1 / (curr_time - self.last_time)
        self.last_time = curr_time
        if (curr_time - self.last_update) > 5.0:
            self.last_update = curr_time
            self.get_logger().info("Computing tracking at %.01f fps" % fps)

    def _init_model(self):
        model = AB3DMOT(max_age=3, min_hits=0)

        return model

    def _create_bbox_message(self, msg, tracking_ids, matched_ids):
        for obj_indx in range(len(matched_ids)):
            if matched_ids[obj_indx] != -1:
                msg.detections[matched_ids[obj_indx]].instance = int(
                    tracking_ids[obj_indx][7]
                )
            else:
                # TODO: replace unsigned int in instance to int
                msg.detections[matched_ids[obj_indx]].instance = -1
        return msg

    def _msg2detections(self, msg):
        detections = []
        for obj_indx in range(len(msg.detections)):
            detection = msg.detections[obj_indx]
            bbox = np.asarray(
                [
                    detection.size_y,
                    detection.size_z,
                    detection.size_x,
                    detection.center_x,
                    detection.center_y,
                    detection.center_z,
                    detection.orientation,
                ]
            )
            detections.append(bbox)
        return np.asarray(detections)


def main(args=None):
    rclpy.init(args=args)

    tracker_publisher = Sort_Tracking()

    rclpy.spin(tracker_publisher)

    tracker_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
