# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import misc.params as params

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        F = np.identity(params.dim_state)
        F[0,3] = params.dt
        F[1,4] = params.dt
        F[2,5] = params.dt

        return F


    def Q(self):
        Q = np.zeros(params.dim_state)
        a = params.dt**3 * params.q / 3
        b = params.dt**2 * params.q / 2
        c = params.dt * params.q
        Q = [[a, 0, 0, b, 0, 0],
             [0, a, 0, 0, b, 0],
             [0, 0, a, 0, 0, b],
             [b, 0, 0, c, 0, 0],
             [0, b, 0, 0, c, 0],
             [0, 0, b, 0, 0, c]
             ]

        return Q


    def predict(self, track):
        track.set_x(self.F() * track.x)
        track.set_P(self.F() * track.P * np.transpose(self.F()) + self.Q())


    def update(self, track, meas):
        H = meas.sensor.get_H(track.x)
        S_inv = np.linalg.inv(self.S(track, meas, H))
        K = track.P * np.transpose(H) * S_inv
        track.set_x(track.x + K * self.gamma(track, meas))
        track.set_P((np.identity(params.dim_state) - K * H) * track.P)

        track.update_attributes(meas)
    

    def gamma(self, track, meas):
        return meas.z - meas.sensor.get_hx(track.x)


    def S(self, track, meas, H):
        return H * track.P * np.transpose(H) + meas.R
