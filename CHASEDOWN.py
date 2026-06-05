# CHAnnel Selection of Electroencophalographic Data Optically With Near-infrared spectroscopy (CHASEDOWN)

import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy.io as sio
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
import math
import pandas as pd
from dataclasses import dataclass, field

sensitivityMatrix = 'dataset/A_temporal.mat'
voxelsPath = 'dataset/voxelCoords_temporal.mat'
hboMatrix = 'dataset/sub-07/Y_HbO_continuous.mat'
eegPath = 'dataset/sub-07/eeg/sub-07_task-eeg_eeg.bdf'
atlasRefPath = 'dataset/colinRefpts.txt'
atlasRefLabelsPath = 'dataset/refpts_labels.txt'

weightFloor = 0.1

@dataclass
class Subject:
  activityMapBOLD: np.ndarray = None
  voxelCoords: np.ndarray = None
  activityMapEEG: np.ndarray = None
  channelNames: np.ndarray = None
  channelCoords: np.ndarray = None
  channelVoxels: dict = field(default_factory=dict)
  channelWeights: np.ndarray = None
  weightedActivity: np.ndarray = None

# Matrix priming to derive spatial activity map
def primeMatrix(subject, sensitivityPath, hboPath, voxelsPath):
  print('priming sensitivity matrix with HbO data')

  # load sensitivity matrix and HbO data
  sensitivityMAT = sio.loadmat(sensitivityPath)
  sensitivityMatrix = sensitivityMAT['Adot']
  print(f'sensitivity matrix shape: {sensitivityMatrix.shape}')

  hboMAT = sio.loadmat(hboPath)
  hboDataRaw = hboMAT['Y_HbO_continuous']
  print(f'HbO data raw shape: {hboDataRaw.shape}')
  # Slice the first 11 rows to isolate just the Oxygenated Hemoglobin (HbO)
  # This turns (33 x Time) into (11 x Time)
  hboData = hboDataRaw[0:11, :]
  print(f'Sliced HbO data shape: {hboData.shape}')

  # 1. Slice the 830nm Forward Model
  # Extracts a 2D matrix of shape (11, 20004)
  A_830 = sensitivityMatrix[:, :, 1]
  print(f'Sliced A (830nm) shape: {A_830.shape}')

  # 2. Invert the Matrix (A_inv)
  # Using the Moore-Penrose pseudoinverse to handle the non-square matrix.
  # A_inv becomes shape (20004, 11)
  A_inv = np.linalg.pinv(A_830)
  print(f'Inverted A (A_inv) shape: {A_inv.shape}')

  # 3. Calculate the Spatial Brain Map (X = A_inv * Y)
  # (20004 x 11) @ (11 x Time) = (20004 x Time)
  X = A_inv @ hboData
  print(f'Final Spatial Map (X) shape: {X.shape}')

  # save to subject class
  subject.activityMapBOLD = X

  #load channel voxel coordinates
  print(f"loading voxel coordinates from {voxelsPath}")
  voxelCoords = sio.loadmat(voxelsPath)
  voxels = voxelCoords['voxel_coords'] # shape: (num_voxels, 3) - x,y,z coordinates of each voxel
  subject.voxelCoords = voxels
  print(f'voxel coordinates shape: {voxels.shape}')

  return subject

# Load EEG channel x time data + channel coordinates
def loadEEG(subject, eegPath):
  print('loading EEG data')

  # 1. Load the raw data FIRST
  raw = mne.io.read_raw_bdf(eegPath, preload=True, verbose=False)

  # 2. Clean the channel names to remove "EEG " prefixes
  mne.rename_channels(raw.info, lambda name: name.replace("EEG ", "").strip())

  # 3. Drop auxiliary and peripheral channels (EMG, ECG, Respiration, etc.)
  drop_channels = [ch for ch in raw.ch_names if ch.upper().startswith(("EXG", "GSR", "STATUS", "TRIG", "AIO", "ERG", "RESP", "PLET", "TEMP"))]
  if drop_channels:
    print(f"Dropping {len(drop_channels)} auxiliary/trigger channels to prevent data leakage...")
    raw.drop_channels(drop_channels)

  # Standard Biosemi 64 to 10-20 mapping
  biosemi_mapping = {
      'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7',
      'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7',
      'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
      'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz', 'A31': 'Pz', 'A32': 'CPz',
      'B1': 'Fp2', 'B2': 'AF8', 'B3': 'AF4', 'B4': 'AFz', 'B5': 'Fz', 'B6': 'F2', 'B7': 'F4', 'B8': 'F6',
      'B9': 'F8', 'B10': 'FT8', 'B11': 'FC6', 'B12': 'FC4', 'B13': 'FC2', 'B14': 'FCz', 'B15': 'Cz', 'B16': 'C2',
      'B17': 'C4', 'B18': 'C6', 'B19': 'T8', 'B20': 'TP8', 'B21': 'CP6', 'B22': 'CP4', 'B23': 'CP2', 'B24': 'P2',
      'B25': 'P4', 'B26': 'P6', 'B27': 'P8', 'B28': 'P10', 'B29': 'PO8', 'B30': 'PO4', 'B31': 'O2', 'B32': 'Fpz'
  }

  # 4. Rename the remaining core channels to standard 10-20
  # We use a try/except or safe mapping in case some channels are already dropped/missing
  safe_mapping = {k: v for k, v in biosemi_mapping.items() if k in raw.ch_names}
  mne.rename_channels(raw.info, safe_mapping)

  # 5. Extract the final matrix
  eegMatrix = raw.get_data()
  subject.activityMapEEG = eegMatrix
  subject.channelNames = raw.ch_names
  print(f'EEG shape (channels x time): {eegMatrix.shape}')

  return subject

# Assign fNIRS voxels to EEG channels based on spatial proximity
def voxelChannelAssn(subject, atlasPath, atlasLabelsPath, kNearest: int = 15):
  print('assigning fNIRS voxels to EEG channels')

  # Extract fiducials from atlas space
  coordsDf = pd.read_csv(atlasPath, sep='\s+', header=None, names=['X', 'Y', 'Z'])

  # Load atlas labels
  atlasLabelsDf = pd.read_csv(atlasLabelsPath, sep='\s+', header=None, names=['label'], dtype={'label': str})
  atlasLabelsDf['label'] = atlasLabelsDf['label'].str.lower()

  refptsDf = pd.concat([atlasLabelsDf, coordsDf], axis=1)
  refptsDf = refptsDf.dropna()

  def get_coord(labels):
    for label in labels:
        row = refptsDf[refptsDf['label'] == label]
        if not row.empty:
            return row[['X', 'Y', 'Z']].values[0]
    raise ValueError(f"Could not find any of {labels} in refpts.txt")
  nzCoord = get_coord(['nz', 'nasion'])
  lpaCoord = get_coord(['al', 'lpa', 'le'])
  rpaCoord = get_coord(['ar', 'rpa', 're'])

  targetFiducials = np.array([nzCoord, lpaCoord, rpaCoord])

  # Extract fiducials from 10-20 template
  montage = mne.channels.make_standard_montage('biosemi64')
  posDict = montage.get_positions()
  templateFiducials = np.array([
    posDict['nasion'],
    posDict['lpa'],
    posDict['rpa']
  ])

  # Extract template EEG channel coordinates
  templateCoords = []
  for ch in subject.channelNames:
    clean_ch = ch.replace("EEG ", "").strip() 
    
    if clean_ch in posDict['ch_pos']:
        templateCoords.append(posDict['ch_pos'][clean_ch])
    else:
        print(f"WARNING: Channel {clean_ch} not found in 10-20 montage.")
        templateCoords.append([0.0, 0.0, 0.0]) 

  templateCoords = np.array(templateCoords)

  # Calculate transformation from template to atlas space using fiducials
  templateCentroid = np.mean(templateFiducials, axis=0)
  targetCentroid = np.mean(targetFiducials, axis=0)

  templateCentered = templateFiducials - templateCentroid
  targetCentered = targetFiducials - targetCentroid

  scaleFactor = np.linalg.norm(targetCentered) / np.linalg.norm(templateCentered)
  templateScaled = templateCentered * scaleFactor

  rotation, rmsd = Rotation.align_vectors(targetCentered, templateScaled)
  print(f'alignment error: {rmsd:.4f} mm')

  # Apply transformation to template EEG channel coordinates
  eegCentered = templateCoords - templateCentroid
  eegScaled = eegCentered * scaleFactor
  eegRotated = rotation.apply(eegScaled)
  eegTransformed = eegRotated + targetCentroid

  subject.channelCoords = eegTransformed
  print(f'EEG channel coordinates shape: {eegTransformed.shape}')

  # Compute distance matrix between EEG channels and fNIRS voxels to make assignments for each channel
  print("calculating euclidean distances for voxel assignment")
  distMatrix = distance.cdist(subject.channelCoords, subject.voxelCoords, metric='euclidean')
  subject.channelVoxels.clear()

  numChannels = subject.channelCoords.shape[0]
  for i in range(numChannels):
    # argsort sorts the distances from smallest to largest and returns the indices
    closestVoxelIDs = np.argsort(distMatrix[i, :])[:kNearest]
    # Key = Channel Index (Int), Value = List of Voxel Indices (List of Ints)
    subject.channelVoxels[i] = closestVoxelIDs.tolist()

  return subject

# Create N x 1 weight vector (N number of EEG channels) based on fNIRS activity levels
def genWeights(subject):
  print ('generating channel weights based on fNIRS activity')

  numChannels = subject.activityMapEEG.shape[0]
  weights = np.zeros((numChannels, 1))

  for i in range(numChannels):
    voxelIDs = subject.channelVoxels[i]
    localFNIRS = subject.activityMapBOLD[voxelIDs, :]
    channelActivity = np.mean(localFNIRS, axis=0)
    peakActivity = np.max(np.abs(channelActivity))
    weights[i, 0] = peakActivity

  minWeight = np.min(weights)
  maxWeight = np.max(weights)

  if maxWeight > minWeight:
    weights = (weights - minWeight) / (maxWeight - minWeight) * (1.0 - weightFloor) + weightFloor
  else:
    weights = np.ones((numChannels, 1))
  
  subject.channelWeights = weights
  print(f'channel weights shape: {weights.shape}')
  print(weights)

  return subject

# Hadamard product to apply weights to EEG data
def applyWeights(subject):
  print('applying weights to EEG data')

  weightedEEG = subject.activityMapEEG * subject.channelWeights
  subject.weightedActivity = weightedEEG

  print(f'weighted EEG shape: {weightedEEG.shape}')
  return subject

# Wrap functions
def getChasedownWeights(eegPath, sensitivityPath, hboPath, voxelsPath, atlasRefPath, atlasRefLabelsPath):
  curSubject = Subject()
  curSubject = primeMatrix(curSubject, sensitivityPath, hboPath, voxelsPath)
  curSubject = loadEEG(curSubject, eegPath)
  curSubject = voxelChannelAssn(curSubject, atlasRefPath, atlasRefLabelsPath, kNearest=15)
  curSubject = genWeights(curSubject)

  print("CHASEDOWN pipeline complete")
  return curSubject.channelWeights

if __name__ == "__main__":
  weights = getChasedownWeights(eegPath, sensitivityMatrix, hboMatrix, voxelsPath, atlasRefPath, atlasRefLabelsPath)
  print("Final channel weights:")
  print(weights)