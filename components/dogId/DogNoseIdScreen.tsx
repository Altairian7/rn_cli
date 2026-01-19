import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, ScrollView, TextInput, PermissionsAndroid, Modal, ActivityIndicator } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import Ionicons from 'react-native-vector-icons/Ionicons';

// Set to your host IP reachable from the device/emulator.
// For the current network (enp5s0f3u1): 10.71.160.212
const BACKEND_URL = 'https://dogid.api.harshie.xyz';

// Normalize base URL to avoid double slashes
const buildUrl = (path: string) => `${BACKEND_URL.replace(/\/+$/, '')}${path}`;

type CaptureResult = {
  saved: boolean;
  capturable: boolean;
  score?: number;
  overlap?: number;
  quality?: {
    passed: boolean;
    laplacian: number;
    tenengrad: number;
    brenner: number;
    contrast: number;
    brightness: number;
    checks: Record<string, boolean>;
  };
  saved_path?: string | null;
  reason?: string | null;
};

interface ResultModalProps {
  type: 'success' | 'error' | null;
  data: any;
  onClose: () => void;
  mode: 'register' | 'identify' | null;
}

const ResultModal: React.FC<ResultModalProps> = ({ type, data, onClose, mode }) => {
  if (!data) return null;

  const isSuccess = type === 'success';
  const isRegisterMode = mode === 'register';

  return (
    <View style={styles.resultContainer}>
      <TouchableOpacity style={styles.resultBackdrop} onPress={onClose} activeOpacity={1} />
      
      <View style={[styles.resultBox, isSuccess ? styles.resultSuccess : styles.resultError]}>
        <View style={styles.resultIconContainer}>
          <View
            style={[
              styles.resultIcon,
              isSuccess ? styles.resultIconSuccess : styles.resultIconError,
            ]}
          >
            <Ionicons
              name={isSuccess ? 'checkmark' : 'close'}
              size={50}
              color="#ffffff"
            />
          </View>
        </View>

        <Text style={styles.resultTitle}>
          {isSuccess
            ? isRegisterMode
              ? '✓ Registration Successful'
              : '✓ Match Found'
            : '⚠ Error'}
        </Text>

        {isSuccess && isRegisterMode && data.dogId && (
          <View style={styles.resultContent}>
            <ResultInfoRow label="Dog ID" value={data.dogId} />
            {data.dogName && data.dogName !== 'N/A' && (
              <ResultInfoRow label="Dog Name" value={data.dogName} />
            )}
            <ResultInfoRow label="Images" value={`${data.images} captured`} />
            <ResultInfoRow label="Time" value={data.timestamp} />
          </View>
        )}

        {isSuccess && !isRegisterMode && data.dogId && (
          <View style={styles.resultContent}>
            <ResultInfoRow label="Identified Dog" value={data.dogId} />
            {data.accuracy && (
              <ResultInfoRow label="Confidence" value={`${data.accuracy}%`} />
            )}
            <ResultInfoRow label="Time" value={data.timestamp} />
          </View>
        )}

        {!isSuccess && data.title && (
          <View style={styles.resultContent}>
            <Text style={styles.resultMessage}>{data.message || data.title}</Text>
          </View>
        )}

        <TouchableOpacity
          style={[styles.resultButton, isSuccess ? styles.resultButtonSuccess : styles.resultButtonError]}
          onPress={onClose}
          activeOpacity={0.85}
        >
          <Text style={styles.resultButtonText}>
            {isSuccess ? '✓ Done' : 'Try Again'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const ResultInfoRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <View style={styles.resultInfoRow}>
    <Text style={styles.resultLabel}>{label}</Text>
    <Text style={styles.resultValue}>{value}</Text>
  </View>
);

export function DogNoseIdScreen() {
  const [scannerOpen, setScannerOpen] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastCapture, setLastCapture] = useState<CaptureResult | null>(null);
  const [currentMode, setCurrentMode] = useState<'register' | 'identify' | null>(null);
  const [dogId, setDogId] = useState('');
  const [dogName, setDogName] = useState('');
  const [location, setLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [imageCount, setImageCount] = useState(0);
  const [showResultModal, setShowResultModal] = useState(false);
  const [resultData, setResultData] = useState<any>(null);
  const [resultType, setResultType] = useState<'success' | 'error' | null>(null);

  const cameraRef = useRef<Camera>(null);
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // Request location permission
  useEffect(() => {
    const requestLocationPerms = async () => {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
          {
            title: 'Location Permission',
            message: 'We need access to your location to improve dog identification accuracy.',
            buttonNeutral: 'Ask Me Later',
            buttonNegative: 'Cancel',
            buttonPositive: 'OK',
          }
        );
        // For now, just set a default location if permission granted
        if (granted === PermissionsAndroid.RESULTS.GRANTED) {
          setLocation({ lat: 0, lon: 0 });
        }
      } catch (err) {
        console.warn('Location error:', err);
      }
    };
    requestLocationPerms();
  }, []);

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

  const postClientLog = async (message: string, meta?: object) => {
    try {
      await fetch(`${BACKEND_URL}/client-log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level: 'info', message, meta }),
      });
    } catch (err) {
      console.warn('client-log failed', err);
    }
  };

  const handleCapture = async () => {
    if (!cameraRef.current || isCapturing) {
      console.log('handleCapture blocked - cameraRef:', !!cameraRef.current, 'isCapturing:', isCapturing);
      return;
    }

    console.log('Starting capture with mode:', currentMode);

    try {
      setIsCapturing(true);
      console.log('Taking photo...');
      const shot = await cameraRef.current.takePhoto({ flash: 'off' });
      console.log('Photo taken, path:', shot.path);

      if (!currentMode) {
        Alert.alert('Error', 'No mode selected. Please try again.');
        setIsCapturing(false);
        return;
      }

      // For register mode, collect multiple images (minimum 2)
      if (currentMode === 'register') {
        console.log('Registering dog - collecting image', imageCount + 1);
        const newImages = [...capturedImages, shot.path];
        setCapturedImages(newImages);
        setImageCount(newImages.length);

        if (newImages.length < 2) {
          Alert.alert(
            'Image Collected',
            `Image ${newImages.length}/2 captured. Please reposition the dog and capture ${2 - newImages.length} more image(s) from different angles.`,
            [{ text: 'Continue', onPress: () => setIsCapturing(false) }]
          );
          return;
        }

        // All 2 images collected, sending to backend
        console.log('All 2 images collected, sending to backend');
        await sendRegisterRequest(newImages);
      } else if (currentMode === 'identify') {
        // For identify, single image is enough
        console.log('Identifying dog with single image');
        await sendIdentifyRequest(shot.path);
      }
    } catch (error) {
      console.error('Capture error:', error);
      Alert.alert('Error', String(error));
      postClientLog('capture_error', { error: String(error) });
    } finally {
      setIsCapturing(false);
    }
  };

  const sendRegisterRequest = async (imagePaths: string[]) => {
    try {
      if (!dogId.trim()) {
        showCustomAlert('Required', 'Dog ID is required for registration', 'error');
        return;
      }

      setIsProcessing(true);

      const form = new FormData();
      
      // Add all 2 images to form - backend expects `files` array
      imagePaths.forEach((path, index) => {
        form.append('files', {
          uri: `file://${path}`,
          name: `nose_${index + 1}.jpg`,
          type: 'image/jpeg',
        } as any);
      });

      form.append('dog_id', dogId);
      if (dogName.trim()) form.append('name', dogName);
      if (location) {
        form.append('lat', location.lat.toString());
        form.append('lon', location.lon.toString());
      }

      const registerUrl = buildUrl('/register');
      console.log('Sending register request with', imagePaths.length, 'images to:', registerUrl);
      
      const res = await fetch(registerUrl, {
        method: 'POST',
        body: form,
        headers: {
          // Don't set Content-Type - let the browser set it with boundary
        }
      });

      console.log('Register response status:', res.status);
      const json = await res.json();
      console.log('Register response data:', json);

      if (!res.ok) {
        const reason = json.detail || json.reason || 'Backend rejected the registration.';
        console.error('Register failed:', reason);
        setIsProcessing(false);
        showCustomAlert('Registration Failed', reason, 'error');
        postClientLog('register_failed', { reason, status: res.status });
        setCapturedImages([]);
        setImageCount(0);
        return;
      }

      setLastCapture(json);
      setScannerOpen(false);
      setCurrentMode(null);
      setCapturedImages([]);
      setImageCount(0);
      
      setResultData({
        dogId,
        dogName: dogName || 'N/A',
        images: imagePaths.length,
        timestamp: new Date().toLocaleString(),
      });
      setResultType('success');
      setShowResultModal(true);
      setIsProcessing(false);
      
      setDogId('');
      setDogName('');
      
      postClientLog('register_success', { dog_id: dogId, images: imagePaths.length });
    } catch (error) {
      console.error('Register error:', error);
      setIsProcessing(false);
      showCustomAlert('Error', String(error), 'error');
      postClientLog('register_error', { error: String(error) });
    }
  };

  const sendIdentifyRequest = async (imagePath: string) => {
    try {
      setIsProcessing(true);

      const form = new FormData();
      form.append('file', {
        uri: `file://${imagePath}`,
        name: 'nose.jpg',
        type: 'image/jpeg',
      } as any);

      if (location) {
        form.append('lat', location.lat.toString());
        form.append('lon', location.lon.toString());
      }

      const identifyUrl = buildUrl('/identify');
      console.log('Sending identify request to:', identifyUrl);
      const res = await fetch(identifyUrl, {
        method: 'POST',
        body: form,
        headers: {
          // Don't set Content-Type - let the browser set it with boundary
        }
      });

      console.log('Identify response status:', res.status);
      const json = await res.json();
      console.log('Identify response data:', json);

      if (!res.ok) {
        const reason = json.detail || json.reason || 'Could not identify the dog.';
        console.error('Identify failed:', reason);
        setIsProcessing(false);
        showCustomAlert('Identification Failed', reason, 'error');
        postClientLog('identify_failed', { reason, status: res.status });
        return;
      }

      setLastCapture(json);
      setScannerOpen(false);
      setCurrentMode(null);
      
      // Extract accuracy/confidence from response if available
      const accuracy = json.score || json.confidence || json.accuracy;
      const matchedDogId = json.dog_id || json.matched_id || 'Unknown';
      
      setResultData({
        dogId: matchedDogId,
        accuracy: accuracy ? (accuracy * 100).toFixed(2) : 'N/A',
        confidence: accuracy ? (accuracy * 100).toFixed(2) : 'N/A',
        timestamp: new Date().toLocaleString(),
        ...json
      });
      setResultType('success');
      setShowResultModal(true);
      setIsProcessing(false);
      
      postClientLog('identify_success', {
        matched_id: matchedDogId,
        accuracy,
        ...json
      });
    } catch (error) {
      console.error('Identify error:', error);
      setIsProcessing(false);
      showCustomAlert('Error', String(error), 'error');
      postClientLog('identify_error', { error: String(error) });
    }
  };

  const showCustomAlert = (title: string, message: string, type: 'success' | 'error') => {
    setResultData({ title, message });
    setResultType(type);
    setShowResultModal(true);
  };

  const handleCloseScanner = () => {
    setScannerOpen(false);
    setCurrentMode(null);
    setCapturedImages([]);
    setImageCount(0);
  };

  useEffect(() => {
    if (!scannerOpen) return;
  }, [scannerOpen]);

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
          isActive={scannerOpen}
          photo={true}
          photoQualityBalance="speed"
          onInitialized={() => console.log('Camera initialized')}
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

        {currentMode === 'register' && (
          <View style={styles.captureCounter}>
            <Text style={styles.counterText}>Images: {imageCount}/2</Text>
          </View>
        )}

        <TouchableOpacity style={styles.closeButton} onPress={handleCloseScanner} activeOpacity={0.9}>
          <Ionicons name="close" size={26} color="#0f172a" />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.shutterButton, isCapturing && styles.shutterDisabled]}
          onPress={() => {
            console.log('Shutter button pressed, currentMode:', currentMode, 'isCapturing:', isCapturing);
            handleCapture();
          }}
          activeOpacity={isCapturing ? 1 : 0.8}
          disabled={isCapturing}
        >
          <View style={styles.shutterInner} />
        </TouchableOpacity>

        {/* Processing Modal */}
        <Modal transparent visible={isProcessing} animationType="fade">
          <View style={styles.processingOverlay}>
            <View style={styles.processingBox}>
              <ActivityIndicator size="large" color="#2563eb" />
              <Text style={styles.processingText}>Processing...</Text>
            </View>
          </View>
        </Modal>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <Text style={styles.pageTitle}>Dog Identification</Text>

        {/* Register Dog Card */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="person-add" size={24} color="#059669" />
            <Text style={styles.cardTitle}>Register New Dog</Text>
          </View>
          <Text style={styles.cardDescription}>
            Register a dog into the system with nose print samples and location data.
          </Text>
          <View style={styles.features}>
            <Text style={styles.featureText}>✓ Multiple nose prints</Text>
            <Text style={styles.featureText}>✓ Dog ID & name</Text>
            <Text style={styles.featureText}>✓ GPS location tracking</Text>
          </View>
          <View style={styles.inputContainer}>
            <Text style={styles.label}>Dog ID *</Text>
            <TextInput
              style={styles.inputField}
              placeholder="Enter dog ID"
              placeholderTextColor="#64748b"
              value={dogId}
              onChangeText={setDogId}
            />
            <Text style={styles.label}>Dog Name (optional)</Text>
            <TextInput
              style={styles.inputField}
              placeholder="Enter dog name"
              placeholderTextColor="#64748b"
              value={dogName}
              onChangeText={setDogName}
            />
            {location && (
              <Text style={styles.locationText}>
                📍 Location: {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
              </Text>
            )}
          </View>
          <TouchableOpacity
            style={[styles.button, { backgroundColor: '#059669' }]}
            onPress={() => {
              console.log('Register button clicked, dogId:', dogId);
              if (!dogId.trim()) {
                Alert.alert('Required', 'Please enter a dog ID');
                return;
              }
              setCurrentMode('register');
              handleStartScan();
            }}
            activeOpacity={0.88}
          >
            <Ionicons name="add-circle" size={20} color="#ffffff" style={{ marginRight: 8 }} />
            <Text style={styles.buttonText}>Register Dog</Text>
          </TouchableOpacity>
        </View>

        {/* Identify Dog Card */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="search" size={24} color="#7c3aed" />
            <Text style={styles.cardTitle}>Identify Dog</Text>
          </View>
          <Text style={styles.cardDescription}>
            Identify a dog by scanning their nose print. Location helps improve accuracy.
          </Text>
          <View style={styles.features}>
            <Text style={styles.featureText}>✓ Single nose print scan</Text>
            <Text style={styles.featureText}>✓ Optional GPS location</Text>
            <Text style={styles.featureText}>✓ Instant match results</Text>
          </View>
          {location && (
            <View style={styles.locationInfo}>
              <Text style={styles.locationText}>
                📍 Location: {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
              </Text>
            </View>
          )}
          <TouchableOpacity
            style={[styles.button, { backgroundColor: '#7c3aed' }]}
            onPress={() => {
              console.log('Identify button clicked');
              setCurrentMode('identify');
              handleStartScan();
            }}
            activeOpacity={0.88}
          >
            <Ionicons name="fingerprint" size={20} color="#ffffff" style={{ marginRight: 8 }} />
            <Text style={styles.buttonText}>Scan to Identify</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* Result Modal on Main Screen */}
      <Modal transparent visible={showResultModal} animationType="slide">
        <ResultModal
          type={resultType}
          data={resultData}
          onClose={() => {
            setShowResultModal(false);
            setResultData(null);
            setResultType(null);
          }}
          mode={currentMode}
        />
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 24,
  },
  pageTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: '#e5e7eb',
    marginBottom: 20,
    marginTop: 8,
  },
  card: {
    borderRadius: 14,
    padding: 18,
    backgroundColor: '#111827',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 4 },
    elevation: 5,
    borderLeftWidth: 4,
    borderLeftColor: '#2563eb',
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#e5e7eb',
    marginLeft: 10,
  },
  cardDescription: {
    fontSize: 13,
    color: '#9ca3af',
    lineHeight: 18,
    marginBottom: 12,
  },
  features: {
    backgroundColor: 'rgba(37, 99, 235, 0.08)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 14,
  },
  featureText: {
    fontSize: 12,
    color: '#cbd5e1',
    marginVertical: 3,
  },
  inputContainer: {
    marginBottom: 12,
    backgroundColor: '#0b1220',
    borderRadius: 8,
    padding: 10,
  },
  label: {
    fontSize: 12,
    fontWeight: '600',
    color: '#94a3b8',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.3,
  },
  inputField: {
    backgroundColor: '#1e293b',
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#334155',
    color: '#e5e7eb',
    fontSize: 13,
  },
  inputPlaceholder: {
    color: '#64748b',
    fontSize: 13,
  },
  locationText: {
    color: '#cbd5e1',
    fontSize: 12,
    marginTop: 8,
    fontWeight: '500',
  },
  locationInfo: {
    backgroundColor: 'rgba(124, 58, 237, 0.1)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 14,
  },
  captureCounter: {
    position: 'absolute',
    top: 60,
    right: 20,
    backgroundColor: 'rgba(37, 99, 235, 0.9)',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    zIndex: 5,
  },
  counterText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 14,
  },
  title: { fontSize: 24, fontWeight: '700', color: '#e5e7eb' },
  subtitle: { marginTop: 6, color: '#9ca3af', lineHeight: 20 },
  steps: { marginTop: 16 },
  stepText: { color: '#cbd5e1', marginTop: 4 },
  button: {
    marginTop: 12,
    borderRadius: 10,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    paddingVertical: 12,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  buttonText: { color: '#ffffff', fontWeight: '700', fontSize: 15 },
  statusPanel: {
    marginTop: 14,
    padding: 12,
    borderRadius: 10,
    backgroundColor: '#0b1220',
  },
  statusLabel: { color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.4 },
  statusValue: { color: '#e5e7eb', fontSize: 13, marginBottom: 6 },
  logPanel: {
    marginTop: 10,
    padding: 12,
    borderRadius: 10,
    backgroundColor: '#0b1220',
    maxHeight: 140,
  },
  logTitle: { color: '#e5e7eb', fontWeight: '600', marginBottom: 6 },
  logList: { maxHeight: 110 },
  logLine: { color: '#cbd5e1', fontSize: 12, marginBottom: 2 },
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
    zIndex: 4,
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
    zIndex: 4,
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
  processingOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  processingBox: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 4 },
    elevation: 10,
  },
  processingText: {
    color: '#e5e7eb',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 16,
  },
  resultContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  resultBackdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  resultBox: {
    backgroundColor: '#111827',
    borderRadius: 24,
    padding: 32,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 8 },
    elevation: 15,
    zIndex: 10,
    maxWidth: 380,
    width: '100%',
  },
  resultSuccess: {
    borderTopWidth: 5,
    borderTopColor: '#10b981',
  },
  resultError: {
    borderTopWidth: 5,
    borderTopColor: '#ef4444',
  },
  resultIconContainer: {
    marginBottom: 24,
  },
  resultIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 8,
  },
  resultIconSuccess: {
    backgroundColor: '#10b981',
  },
  resultIconError: {
    backgroundColor: '#ef4444',
  },
  resultTitle: {
    fontSize: 24,
    fontWeight: '900',
    color: '#f1f5f9',
    marginBottom: 20,
    textAlign: 'center',
    letterSpacing: 0.3,
  },
  resultContent: {
    width: '100%',
    marginBottom: 28,
    backgroundColor: 'rgba(30, 41, 59, 0.8)',
    borderRadius: 16,
    padding: 20,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  resultInfoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(51, 65, 85, 0.5)',
  },
  resultLabel: {
    fontSize: 12,
    fontWeight: '700',
    color: '#cbd5e1',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  resultValue: {
    fontSize: 15,
    fontWeight: '800',
    color: '#f1f5f9',
    marginLeft: 8,
    flex: 1,
    textAlign: 'right',
  },
  resultMessage: {
    fontSize: 15,
    color: '#e2e8f0',
    textAlign: 'center',
    lineHeight: 24,
    fontWeight: '500',
  },
  resultButton: {
    width: '100%',
    paddingVertical: 16,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
    elevation: 6,
  },
  resultButtonSuccess: {
    backgroundColor: '#10b981',
  },
  resultButtonError: {
    backgroundColor: '#ef4444',
  },
  resultButtonText: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
});