import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import Ionicons from 'react-native-vector-icons/Ionicons';

type RegistryResult = {
  found: boolean;
  name?: string;
  registryId?: string;
};

export function DogNoseIdScreen() {
  const [scannerOpen, setScannerOpen] = useState(false);
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<RegistryResult | null>(null);

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
    setResult(null);
  };

  const handleCaptureNose = async () => {
    if (!cameraRef.current) return;

    try {
      await cameraRef.current.takePhoto({ flash: 'off' });
      setScannerOpen(false);
      setChecking(true);

      // Simulated registry lookup
      setTimeout(() => {
        setChecking(false);
        setResult({
          found: true,
          name: 'Registered Pup',
          registryId: `DOG-${Date.now().toString().slice(-6)}`,
        });
        Alert.alert('Match found', 'This nose print is in the registry.');
      }, 900);
    } catch (error) {
      console.error(error);
      Alert.alert('Capture failed', 'Could not capture the nose print.');
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

        <View style={styles.cameraOverlay}>
          <Text style={styles.overlayTitle}>Align the dog nose inside the frame</Text>

          <TouchableOpacity style={styles.captureButton} onPress={handleCaptureNose} activeOpacity={0.9}>
            <Ionicons name="paw" size={26} color="#111827" style={{ marginRight: 8 }} />
            <Text style={styles.captureText}>Capture nose</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.closeButton} onPress={handleCloseScanner}>
            <Ionicons name="close" size={26} color="#111827" />
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.card}>
        <Text style={styles.title}>Unique Dog Identification</Text>
        <Text style={styles.subtitle}>Scan a dog nose print to verify if they are already in the registry.</Text>

        <View style={styles.steps}>
          <Text style={styles.stepText}>1. Position the dog comfortably.</Text>
          <Text style={styles.stepText}>2. Tap the paw button to open the camera.</Text>
          <Text style={styles.stepText}>3. Center the nose print and capture.</Text>
        </View>

        <TouchableOpacity style={styles.button} onPress={handleStartScan} activeOpacity={0.88}>
          <Ionicons name="paw" size={22} color="#ffffff" style={{ marginRight: 10 }} />
          <Text style={styles.buttonText}>Scan nose</Text>
        </TouchableOpacity>

        <Text style={styles.footerText}>
          We use nose textures only to check the registry; nothing is stored on device.
        </Text>
      </View>

      {checking && (
        <View style={styles.statusCard}>
          <ActivityIndicator color="#2563eb" size="small" />
          <Text style={styles.statusText}>Checking registry…</Text>
        </View>
      )}

      {result && !checking && (
        <View style={styles.statusCard}>
          <Ionicons
            name={result.found ? 'checkmark-circle' : 'alert-circle'}
            size={22}
            color={result.found ? '#16a34a' : '#f59e0b'}
            style={{ marginRight: 8 }}
          />
          <View>
            <Text style={styles.statusTitle}>{result.found ? 'Registry match' : 'No match found'}</Text>
            {result.found && (
              <Text style={styles.statusDetail}>
                {result.name} • ID {result.registryId}
              </Text>
            )}
          </View>
        </View>
      )}
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
  footerText: { marginTop: 12, textAlign: 'center', color: '#94a3b8', lineHeight: 18 },
  statusCard: {
    marginTop: 14,
    borderRadius: 12,
    backgroundColor: '#111827',
    padding: 14,
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusText: { color: '#cbd5e1', marginLeft: 10 },
  statusTitle: { color: '#e5e7eb', fontWeight: '600' },
  statusDetail: { color: '#94a3b8', marginTop: 2 },
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  cameraOverlay: {
    position: 'absolute',
    bottom: 48,
    left: 20,
    right: 20,
    alignItems: 'center',
    gap: 14,
  },
  overlayTitle: {
    color: '#f8fafc',
    textAlign: 'center',
    marginBottom: 6,
    fontWeight: '600',
  },
  captureButton: {
    backgroundColor: '#f8fafc',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 30,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureText: { color: '#111827', fontWeight: '700', fontSize: 16 },
  closeButton: {
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderRadius: 20,
    padding: 10,
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
});
