""" Spike Sorter
"""

# Author: Guillem Boada <guillemboada@hotmail.com>

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin

import os
import numpy as np
from scipy import signal
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.mixture import GaussianMixture
import scipy
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from kneed.data_generator import DataGenerator
from kneed.knee_locator import KneeLocator
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
import warnings
import brpylib  # BlackRock library

warnings.filterwarnings("ignore")

def NEO(x):
    """ Calculate Nonlinear energy operator (NEO) of a signal.
        The first and last values are added zeros, so that the
        output and input shape are equal.

    Args:
        x (numpy.array): (n_channels, n_samples) or (n_samples) Signal

    Returns:
        numpy.array: NEO
    """

    # for 1D array (n_samples)
    if len(x.shape) == 1:
        x_pad = np.pad(x, pad_width=2, mode='constant')
        product = x_pad[:-2] * x_pad[2:]

        neo = x[1:-1]**2 - product[2:-2]  # NEO formula
        neo_full = np.pad(neo, pad_width=1, mode='constant')

    # for 2D array (n_channels, n_samples)
    else:
        x_pad = np.pad(x, pad_width=((0, 0), (2, 2)), mode='constant')
        product = x_pad[:, :-2] * x_pad[:, 2:]

        neo = x[:, 1:-1]**2 - product[:, 2:-2]  # NEO formula
        neo_full = np.pad(neo, pad_width=((0, 0), (1, 1)), mode='constant')

    return neo_full


class Waveformer(TransformerMixin):
    '''Waveformer.

    Given a raw trace, it extracts waveforms following the common
    pipeline consisting of filtering, emphasization, thresholding,
    waveform extraction and alignment.

    References
    * Gibson, S., Judy, J. W., & Marković, D. (2011). Spike sorting: The
    first step in decoding the brain: The first step in decoding the brain.
    IEEE Signal Processing Magazine, 29(1), 124–143.
    https://doi.org/10.1109/MSP.2011.941880
    '''

    def __init__(self, filter_type='bandpass',
                 filter_freqs=[500, 4000], filter_order=5,
                 method_emphasize='absolute value', factor=5,
                 min_distance_between_crossings=.001,
                 wide_window_limits=(35, 65),
                 narrow_window_limits=(25, 35),
                 method_align='min amplitude',
                 clean=True,
                 visualize=False,
                 verbose=False):

        self.filter_type = filter_type
        self.filter_freqs = filter_freqs
        self.filter_order = filter_order
        self.method_emphasize = method_emphasize
        self.factor = factor
        self.min_distance_between_crossings = min_distance_between_crossings
        self.wide_window_limits = wide_window_limits
        self.narrow_window_limits = narrow_window_limits
        self.method_align = method_align
        self.clean = clean
        self.visualize = visualize
        self.verbose = verbose

    def transform(self, X, fs=30000, *_):
        self.X = X
        self.fs = fs
        self.X_filtered_ = self._filter()
        self.X_emphasized_ = self._emphasize()
        self.threshold_ = self._set_threshold()
        self.detection_points_ = self._threshold_signal()
        self.wide_windows_, self.wide_windows_starts = self._extract_windows(
            wide_or_narrow='wide')
        self.reference_points_, self.timestamps_ = self._align()
        self.waveforms_, _ = self._extract_windows(wide_or_narrow='narrow')
        if self.clean==True:
            self.waveforms_, self.timestamps_, self.idx_outliers_ = self._discard_outliers()
        if self.visualize==True:
            self._plot_waveforms()
        # Stack the timestamps and waveforms- (n_waveforms, n_samples + 1),
        # the +1 is the 1st column for the timestamps
        result = np.hstack(
            (np.reshape(self.timestamps_, (self.timestamps_.size, 1)), self.waveforms_))
        return result

    def fit(self, *_):
        return self

    def _filter(self):
        """ Filter signal

        Args:
            X (numpy.array): signal
            filter_type (str): filter type {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
            filter_freqs (numpy.array, list): cutoff frequencies
            filter_order (numpy.array): filter order

        Returns:
            numpy.array: filtered signal
        """    	
        nyquist_freq = self.fs / 2
        filter_freqs_normalized = np.array(self.filter_freqs) / nyquist_freq
        b, a = signal.butter(
            self.filter_order, filter_freqs_normalized, btype=self.filter_type)
        return signal.filtfilt(b, a, self.X)

    def _emphasize(self):
        if self.method_emphasize == 'absolute value':
            return np.abs(self.X_filtered_)
        if self.method_emphasize == 'NEO':
            return NEO(self.X_filtered_)

    def _set_threshold(self):
        # Set heuristic threshold
        # for absolute [Quiroga, 2004]. Recommended constant = 5
        # for NEO [Malik, 2016]. Recommended constant = 10

        if self.method_emphasize == 'absolute value':
            threshold = self.factor * \
                np.median(np.abs(self.X_filtered_) / 0.6745)
        if self.method_emphasize == 'NEO':
            threshold = self.factor * 1 / N * np.sum(NEO(self.X_filtered_))
        if self.verbose == True:
            print(f'Threshold: {threshold} uV')
        return threshold

    def _threshold_signal(self):
        """ Given a signal and a threshold, it returns the timestimaps of the upward threshold crossings

        Args:
            X (numpy.array): signal
            threshold (numpy.array): threshold
            min_distance_between_crossings (numpy.array): used to delete crossings too close in time

        Returns:
            numpy.array: upward crossings samplestamps
        """
        # Set as 1 all sample above threshold
        X_above_threshold = (self.X_emphasized_ > self.threshold_).astype(int)
        # Find the crossing by taking its difference
        X_diff = np.diff(X_above_threshold)
        # Find where the upward crossings happen
        upward_crossings = np.where(X_diff == 1)[0]
        # Calculate the distance between upward crossings
        distance_crossings = -np.diff(upward_crossings[::-1])
        # Make mask to indentfy detections which happen just afer another detection
        bool_crossings = np.append(True, (distance_crossings > (
            self.min_distance_between_crossings * self.fs))[::-1])
        # Mask them out because they are multiple detections of the same waveform
        accepted_crossings = upward_crossings[bool_crossings]
        return accepted_crossings

    def _extract_windows(self, wide_or_narrow):
        """ Given a signal and detection points in samples, it returns signal windows with respect to those points.

        Args:
            X (numpy.array): signal
            detections (numpy.array): sample points to be used as reference
            window_limits (numpy.array, list): amount of samples that are taken from the left and right

        Returns:
            2D numpy.array: (n_waveforms, n_samples) stacked waveforms
            numpy.array: window start points
        """
        # Select limits depending on ``size``
        if wide_or_narrow == 'wide':
            limits = self.wide_window_limits
            center_points = self.detection_points_
        if wide_or_narrow == 'narrow':
            limits = self.narrow_window_limits
            center_points = self.reference_points_
        # General procedure to extract windows
        w_width = sum(limits)
        w_start, w_end = center_points - limits[0], center_points + limits[1]
        n_waveforms = center_points.size
        w = np.zeros((n_waveforms, w_width))
        for i in range(n_waveforms):
            if (w_start[i] > 0) and (w_end[i] <= self.X.size):
                w[i] = self.X_filtered_[w_start[i]:w_end[i]]
        return w, w_start

    def _align(self):
        if self.method_align == 'min amplitude':
            reference_points = np.argmin(
                self.wide_windows_, axis=1) + self.wide_windows_starts
        if self.method_align == 'NEO':
            reference_points = np.argmax(
                NEO(self.wide_windows_), axis=1) + self.wide_windows_starts
        timestamps = reference_points / self.fs
        return reference_points, timestamps
    
    def _discard_outliers(self, variance_threshold=5000):
        # Use waveform variances
        variances = np.var(self.waveforms_, axis=1)
        # Detect outliers
        idx_outliers = np.where(variances > variance_threshold)[0]
        # Discard them
        waveforms_clean = np.delete(self.waveforms_, tuple(idx_outliers), axis=0)
        timestamps_clean = np.delete(self.timestamps_, tuple(idx_outliers), axis=0)

        return waveforms_clean, timestamps_clean, idx_outliers
    
    def _plot_waveforms(self):
        """ Visualize overlapped waveforms.

        Args:
            waveforms (2D numpy.array): (n_waveforms, n_samples) stacked waveforms
            fs (numpy.array): sampling frequency
            window_limits (numpy.array, list): waveform expansion to the left and right
        """
        time = np.arange(self.waveforms_[0].size)/self.fs*1000
        for waveform in self.waveforms_:
            plt.plot(time, waveform, c='k', alpha=.05)
        plt.xlim([0, sum(self.narrow_window_limits)/self.fs*1000])
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [uV]')  


class WaveformSorter(BaseEstimator, ClassifierMixin):
    '''Waveform Sorter.

    Given a collection of waveforms features, it applies a clustering
    algorithm (mixture of Gaussians) with an automatically selected
    number of cluster to assign a label to each waveform.
    '''

    def __init__(self, max_components=3, visualize=False, n_iter=10, random_state=None):

        self.random_state = random_state
        self.max_components = max_components
        self.n_iter = n_iter
        self.visualize = visualize

    def fit(self, X, y=None):
        """
        Determines means and covariances of the Gaussians.
        """

        # Discard outliers
        self.X_clean_, self.idx_outliers_ = self._discard_outliers(X)
        # Check if enough instances to cluster
        if X.shape[0] < 100:
            raise Exception('Not enough waveforms to cluster (less than 100).')
        else:
            # Take timestamps and PCs separately
            self.timestamps_ = self.X_clean_[:, 0]
            self.pcs_ = self.X_clean_[:, 1:]
            # Select k and fit clusterer
            self.n_components_ = self._select_k()
            self.gmm = GaussianMixture(
                n_components=self.n_components_, random_state=self.random_state).fit(self.pcs_)
        return self

    def predict(self, X, y=None):
        X_clean, _ = self._discard_outliers(X)
        if X_clean.shape[0] < 100:
            self.labels_ = [-1 for i in range(X_clean.shape[0])]
        else:
            # Take timestamps and PCs separately
            self.timestamps_ = self.X_clean_[:, 0]
            self.pcs_ = self.X_clean_[:, 1:]
            # Predict labels
            self.labels_ = self.gmm.predict(self.pcs_)
            
        # Stack the timestamps and labels (n_waveforms, 2)
        Y = np.vstack((self.timestamps_, self.labels_)).T
            
        if self.visualize==True:
            self._plot_clusters()
            self._correlogram()
            self._plot_correlogram()
            
        return Y

    def score(self, X, y=None):
        # Just an example
        return(sum(self.predict(X)))

    def _discard_outliers(self, X):
        # Take timestamps and PCs separately
        timestamps = X[:, 0]
        pcs = X[:, 1:]
        # Use distance between samples in the PC space to identify outliers
        distances = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Calculate Euclidian distance between 2 points
            distances[i] = (np.linalg.norm(pcs[0]-pcs[i]))
        # Detect outliers
        idx_outliers = np.where(distances > 1000)[0]
        # Discard them
        X_clean = np.delete(X, tuple(idx_outliers), axis=0)
        return X_clean, idx_outliers

    def _select_k(self):
        # Take only the PCs
        pcs = self.X_clean_[:, 1:]
        BIC = []
        for i in range(self.n_iter):
            # Train model with several k, until max_cluster
            models = [GaussianMixture(n_components=k, random_state=self.random_state).fit(pcs)
                      for k in range(1, self.max_components+1)]
            # Compute the BIC for each model
            BIC.append([model.bic(pcs) for model in models])
        # Decide k using BIC
        BIC_diff = np.diff(np.mean(np.array(BIC), axis=0))
        if np.min(BIC_diff) < 0:
            k = np.argmin(BIC_diff)+2
        else:
            k = 1
        return k

    def _plot_clusters(self):
        pcs_list = [f'PC{i+1}' for i in range(3)]
        df = pd.DataFrame(self.pcs_, columns=pcs_list)
        if list(set(self.labels_))[0]==-1: # checks that all elements are -1, i.e. a channel which was not clustered 
            pp = sns.pairplot(df, markers='+', aspect=1, diag_kind='hist',                          
                              plot_kws={'color':'k', 'edgecolor':None, 'alpha':.5},
                              diag_kws={'color':'k'})  
        else:
            df['label'] = self.labels_
            pp = sns.pairplot(df, vars=pcs_list, hue='label', markers='+', aspect=1, diag_kind='hist',
                              plot_kws={'edgecolor':None, 'alpha':.5})
            pp._legend.remove()
        for i, j in zip(*np.triu_indices_from(pp.axes, 1)):
            pp.axes[i, j].set_visible(False)
        plt.show()
        
    def _correlogram(self):
        '''Calculate cross correlogram
          ccg = correlogram(t, assignment, binsize, maxlag) calculates the 
          cross- and autocorrelograms for all pairs of clusters with input
              t               spike times             #spikes x 1
              assignment      cluster assignments     #spikes x 1
              binsize         bin size in ccg         scalar
              maxlag          maximal lag             scalar

           and output
              ccg             computed correlograms   #bins x #clusters x
                                                                      #clusters
              bins            bin times relative to center    #bins x 1
        '''
        binsize = 0.001
        maxlag = 0.020
        nbins = int(np.round(maxlag/binsize))
        bins = np.arange(-nbins, nbins+1)
        ccg_list = []
        bins_list = []
        if list(set(self.labels_))[0]==-1: # if the channel wasn't clustered (marked with labels=-1), return a ccg of zeros
            ccg_list.append(np.zeros((nbins*2+1, 2, 2)))

        else:

            t = np.array(self.timestamps_)
            assignment = np.array(self.labels_)

            # we need the spike times to be sorted
            idx = np.argsort(t)
            t = np.sort(t)

            assignment = assignment[idx]
            K = np.max(assignment)+1
            ccg = np.zeros([2 * nbins + 1, K, K])
            N = t.size
            j = 0

            # for each timestamp only consider the timestamps that are within the maxlag window around it 
            for i in range(N):
                # iterate j back
                while j > 0 and t[j] > t[i] - maxlag:
                    j -= 1
                # iterate through all timestamps that are 'maxlag' ms centered around t[i]
                while j < N-1 and t[j + 1] < t[i] + maxlag:
                    j += 1
                    if i != j: # don't compute the zero time lag bin for the auto-correlation

                        # the offset of both timestamps normed by the binsize gives the bin into which the found 
                        # correlation counts
                        off = np.abs((np.round((t[i] - t[j]) / binsize)))
                        ccg[int(nbins+off), assignment[i], assignment[j]] += 1
                        ccg[int(nbins-off), assignment[i], assignment[j]] += 1
            
        self.ccg = ccg
        self.bins = bins
        return self

    def _plot_correlogram(self):
        '''Plot cross-correlograms of all pairs.
           plotCCG(ccg,bins) plots a matrix of cross(auto)-correlograms for
           all pairs of clusters. Inputs are:
              ccg     array of cross correlograms           #bins x #clusters x #clusters
              bins    array with bin timings                #nbins x 0
        '''
        colors = sns.color_palette()

        if (self.ccg==0).all():
            return

        plt.clf
        bg = 0.7*np.ones(3)
        K = self.ccg.shape[1] 
        for ix in range(K):
            for jx in range(K):
                if jx<=ix:
                    ax = plt.subplot(K, K, K*ix+jx+1, facecolor=bg)
                    if ix == jx:
                        ax.bar(self.bins, self.ccg[:, ix, jx], width=1, facecolor=colors[ix], edgecolor=colors[ix])
                    else:
                        ax.bar(self.bins, self.ccg[:, ix, jx], width=1, facecolor='k')
                    ax.axis('on')
                    ax.set_xlim(1.2 * self.bins[[0,-1]])
                    ylim = np.array(list(ax.get_ylim()))
                    ax.set_ylim(np.array([0, 1.2]) * ylim)
                    ax.set_yticks([])
                    if ix != jx:
                        ax.plot(0, 0, '*', c=colors[jx])
                    if ix != K-1:
                        ax.set_xticks([])
                    if ix == K-1:
                        ax.set_xlabel(f'Cluster {jx+1} (ms)')
                    if jx == 0:
                        ax.set_ylabel(f'Cluster {ix+1}')
        plt.show()
