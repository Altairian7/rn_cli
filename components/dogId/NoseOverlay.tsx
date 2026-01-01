import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';

export type NoseOverlayProps = {
  alignmentReady: boolean;
  sharpReady: boolean;
  lockProgress: number;
  readyToCapture: boolean;
  isCapturing: boolean;
  autoCaptureTriggered: boolean;
  onCapture: () => void;
  onClose: () => void;
};

export function NoseOverlay(props: NoseOverlayProps) {
  const {
    alignmentReady,
    sharpReady,
    lockProgress,
    readyToCapture,
    isCapturing,
    autoCaptureTriggered,
    onCapture,
    onClose,
  } = props;

  return (
    <View style={styles.overlayContainer}>
      <Text style={styles.overlayTitle}>Align the dog nose inside the frame</Text>

      <View style={styles.lockRow}>
        <View style={[styles.lockPill, alignmentReady ? styles.lockOn : styles.lockOff]}>
          <Ionicons
            name={alignmentReady ? 'checkmark-circle' : 'ellipse-outline'}
            size={18}
            color={alignmentReady ? '#16a34a' : '#f8fafc'}
          />
          <Text style={styles.lockText}>Centered</Text>
        </View>
        <View style={[styles.lockPill, sharpReady ? styles.lockOn : styles.lockOff]}>
          <Ionicons
            name={sharpReady ? 'checkmark-circle' : 'ellipse-outline'}
            size={18}
            color={sharpReady ? '#16a34a' : '#f8fafc'}
          />
          <Text style={styles.lockText}>Sharp</Text>
        </View>
      </View>

      <View style={styles.progressTrack}>
        <View style={[styles.progressFill, { width: `${lockProgress}%` }]} />
      </View>
      <Text style={styles.progressHint}>
        {readyToCapture
          ? autoCaptureTriggered
            ? 'Locked. Auto capture in progress.'
            : 'Locked. Auto capturing shortly.'
          : 'Hold steady until centered and sharp.'}
      </Text>

      <TouchableOpacity
        style={[styles.captureButton, (!readyToCapture || isCapturing) && styles.captureButtonDisabled]}
        onPress={onCapture}
        activeOpacity={readyToCapture && !isCapturing ? 0.9 : 1}
        disabled={!readyToCapture || isCapturing}
      >
        {isCapturing ? (
          <ActivityIndicator color="#111827" />
        ) : (
          <>
            <Ionicons name="paw" size={26} color="#111827" style={{ marginRight: 8 }} />
            <Text style={styles.captureText}>{autoCaptureTriggered ? 'Capturing…' : 'Capture now'}</Text>
          </>
        )}
      </TouchableOpacity>

      <TouchableOpacity style={styles.closeButton} onPress={onClose}>
        <Ionicons name="close" size={26} color="#111827" />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  overlayContainer: {
    position: 'absolute',
    bottom: 48,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  overlayTitle: {
    color: '#f8fafc',
    textAlign: 'center',
    marginBottom: 6,
    fontWeight: '600',
  },
  lockRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 10,
  },
  lockPill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    marginHorizontal: 5,
  },
  lockOn: {
    backgroundColor: 'rgba(22, 163, 74, 0.18)',
    borderColor: '#16a34a',
    borderWidth: 1,
  },
  lockOff: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderColor: 'rgba(255,255,255,0.18)',
    borderWidth: 1,
  },
  lockText: { color: '#f8fafc', marginLeft: 8, fontWeight: '600' },
  progressTrack: {
    marginTop: 12,
    width: '100%',
    height: 8,
    borderRadius: 6,
    backgroundColor: 'rgba(255,255,255,0.18)',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#22c55e',
  },
  progressHint: {
    marginTop: 8,
    color: '#f8fafc',
    textAlign: 'center',
    fontWeight: '600',
  },
  captureButton: {
    marginTop: 12,
    backgroundColor: '#f8fafc',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 30,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 160,
  },
  captureButtonDisabled: {
    opacity: 0.6,
  },
  captureText: { color: '#111827', fontWeight: '700', fontSize: 16 },
  closeButton: {
    marginTop: 12,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderRadius: 20,
    padding: 10,
  },
});
