#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhangping
# Create at : 2018.1.8

class PlannerBase(object):
    """This is Base class for action planner, it will be inherited."""
    def __init__(self):
        print "Base Planner online..."

    def run(self, features):
        """Run action planner, this is a demo. So just return 0.
        return: action angle.
        """
        return 0
