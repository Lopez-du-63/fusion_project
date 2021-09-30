# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
        N = len(track_list)
        M = len(meas_list)

        self.association_matrix = np.inf * np.ones((N,M)) # reset matrix
        self.unassigned_tracks = list(range(N)) # reset lists
        self.unassigned_meas = list(range(M))

        for i in range(N): 
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                dist = self.MHD(track, meas, KF)
                if self.gating(dist, meas.sensor):
                    self.association_matrix[i,j] = dist

                
    def get_closest_track_and_meas(self):
        index = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        if self.association_matrix[index] == np.inf:
            return np.nan, np.nan
        else:
            self.association_matrix = np.delete(self.association_matrix, index[0], 0)
            self.association_matrix = np.delete(self.association_matrix, index[1], 1)

            update_track = self.unassigned_tracks[index[0]]
            update_meas = self.unassigned_meas[index[1]]

            self.unassigned_tracks = np.delete(self.unassigned_tracks, index[0])
            self.unassigned_meas = np.delete(self.unassigned_meas, index[1])
            return update_track, update_meas
        

    def gating(self, MHD, sensor): 
        #return True
        return True if MHD < chi2.ppf(params.gating_threshold, sensor.dim_meas) else False

        
    def MHD(self, track, meas, KF):
        H = meas.sensor.get_H(track.x)
        gamma = KF.gamma(track, meas)
        return np.transpose(gamma) * np.linalg.inv(KF.S(track, meas, H)) * gamma
        
        
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)