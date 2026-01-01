import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import Ionicons from 'react-native-vector-icons/Ionicons';

export function DogNoseIdScreen() {
  const [scannerOpen, setScannerOpen] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);

  const cameraRef = useRef<Camera>(null);
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const handleStartScan = async () => {
    if (!hasPermission) {
      const granted = await requestPermission();
      if (!granted) {
        Alert.alert('Camera needed', 'Allow camera to scan a nose print.');
        return;
      }
    }

    if (device == null) {
      Alert.alert('No camera found', 'We could not detect a camera device.');
      return;
    }

    setScannerOpen(true);
  };

  const handleCapture = async () => {
    if (!cameraRef.current || isCapturing) return;

    try {
      setIsCapturing(true);
      const shot = await cameraRef.current.takePhoto({ flash: 'off' });
      setScannerOpen(false);
      Alert.alert('Nose photo saved', `Captured at ${shot.path}`);
    } catch (error) {
      console.error(error);
      Alert.alert('Capture failed', 'Could not capture the nose.');
    } finally {
      setIsCapturing(false);
    }
  };

  const handleCloseScanner = () => {
    setScannerOpen(false);
  };

  if (scannerOpen) {
    if (!hasPermission) {
      return (
        <View style={styles.cameraFallback}>
          <Text style={styles.message}>Camera permission is required to scan.</Text>
          <TouchableOpacity style={styles.secondaryButton} onPress={requestPermission}>
            <Text style={styles.secondaryButtonText}>Grant permission</Text>
          </TouchableOpacity>
        </View>
      );
    }

    if (device == null) {
      return (
        <View style={styles.cameraFallback}>
          <Text style={styles.message}>No camera device detected.</Text>
        </View>
      );
    }

    return (
      <View style={styles.cameraContainer}>
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
          photoQualityBalance="speed"
        />

        <View style={styles.maskContainer} pointerEvents="none">
          <View style={[styles.dimBlock, styles.dimTop]} />
          <View style={styles.dimRow}>
            <View style={[styles.dimBlock, styles.dimSide]} />
            <View style={styles.noseFrame} />
            <View style={[styles.dimBlock, styles.dimSide]} />
          </View>
          <View style={[styles.dimBlock, styles.dimBottom]} />
        </View>

        <TouchableOpacity style={styles.closeButton} onPress={handleCloseScanner} activeOpacity={0.9}>
          <Ionicons name="close" size={26} color="#0f172a" />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.shutterButton, isCapturing && styles.shutterDisabled]}
          onPress={handleCapture}
          activeOpacity={isCapturing ? 1 : 0.8}
          disabled={isCapturing}
        >
          <View style={styles.shutterInner} />
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.card}>
        <Text style={styles.title}>Unique Dog Identification</Text>
        <Text style={styles.subtitle}>Open the camera and align the nose inside the dotted frame to capture a clear print.</Text>

        <View style={styles.steps}>
          <Text style={styles.stepText}>1. Position the dog comfortably.</Text>
          <Text style={styles.stepText}>2. Tap the paw button to open the camera.</Text>
          <Text style={styles.stepText}>3. Center the nose print and tap capture.</Text>
        </View>

        <TouchableOpacity style={styles.button} onPress={handleStartScan} activeOpacity={0.88}>
          <Ionicons name="paw" size={22} color="#ffffff" style={{ marginRight: 10 }} />
          <Text style={styles.buttonText}>Scan nose</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 18,
    justifyContent: 'center',
    backgroundColor: '#0f172a',
  },
  card: {
    borderRadius: 16,
    padding: 22,
    backgroundColor: '#111827',
    shadowColor: '#000',
    shadowOpacity: 0.18,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 },
    elevation: 6,
  },
  title: { fontSize: 24, fontWeight: '700', color: '#e5e7eb' },
  subtitle: { marginTop: 6, color: '#9ca3af', lineHeight: 20 },
  steps: { marginTop: 16 },
  stepText: { color: '#cbd5e1', marginTop: 4 },
  button: {
    marginTop: 18,
    borderRadius: 12,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    paddingVertical: 14,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  buttonText: { color: '#ffffff', fontWeight: '700', fontSize: 16 },
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  maskContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
  },
  dimBlock: {
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  dimTop: {
    flex: 1,
  },
  dimRow: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dimSide: {
    flex: 1,
  },
  dimBottom: {
    flex: 1.2,
  },
  noseFrame: {
    width: 238,
    height: 192,
    borderWidth: 3,
    borderStyle: 'dashed',
    borderColor: 'rgba(255,255,255,0.9)',
    borderTopLeftRadius: 120,
    borderTopRightRadius: 120,
    borderBottomLeftRadius: 178,
    borderBottomRightRadius: 178,
    backgroundColor: 'rgba(0,0,0,0.08)',
    alignItems: 'center',
    justifyContent: 'flex-end',
    paddingBottom: 32,
  },
  cameraFallback: {
    flex: 1,
    backgroundColor: '#0f172a',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  message: { color: '#e5e7eb', textAlign: 'center', marginBottom: 14 },
  secondaryButton: {
    backgroundColor: '#2563eb',
    borderRadius: 10,
    paddingHorizontal: 18,
    paddingVertical: 12,
  },
  secondaryButtonText: { color: '#ffffff', fontWeight: '600' },
  closeButton: {
    position: 'absolute',
    top: 22,
    left: 18,
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderRadius: 18,
    padding: 8,
  },
  shutterButton: {
    position: 'absolute',
    bottom: 28,
    alignSelf: 'center',
    width: 82,
    height: 82,
    borderRadius: 50,
    backgroundColor: 'rgba(255,255,255,0.9)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 4,
    borderColor: 'rgba(0,0,0,0.25)',
  },
  shutterInner: {
    width: 62,
    height: 62,
    borderRadius: 40,
    backgroundColor: '#ffffff',
  },
  shutterDisabled: {
    opacity: 0.6,
  },
});
