import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, ScrollView } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import Ionicons from 'react-native-vector-icons/Ionicons';

// Set to your host IP reachable from the device/emulator.
// For the current network (enp5s0f3u1): 10.71.160.212
const BACKEND_URL = 'http://10.71.160.212:8000';

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

export function DogNoseIdScreen() {
  const [scannerOpen, setScannerOpen] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [backendLogs, setBackendLogs] = useState<{ ts: number; source: string; message: string }[]>([]);
  const [connection, setConnection] = useState<'unknown' | 'ok' | 'down'>('unknown');
  const [lastCapture, setLastCapture] = useState<CaptureResult | null>(null);

  const cameraRef = useRef<Camera>(null);
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // Background connectivity check even when scanner is closed
  useEffect(() => {
    let mounted = true;
    const run = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/health`);
        if (!mounted) return;
        setConnection(res.ok ? 'ok' : 'down');
      } catch (e) {
        if (mounted) setConnection('down');
      }
    };
    run();
    const id = setInterval(run, 6000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
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
    setStatus('Scanner open');
    pingBackend();
  };

  const pingBackend = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/health`);
      if (res.ok) {
        setConnection('ok');
        return;
      }
    } catch (e) {
      // ignore
    }
    setConnection('down');
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

  const pollBackendLogs = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/logs`);
      const json = await res.json();
      const logs = Array.isArray(json.logs) ? json.logs : [];
      setBackendLogs(logs.slice(-5));
    } catch (err) {
      // ignore
    }
  };

  const handleCapture = async () => {
    if (!cameraRef.current || isCapturing) return;

    try {
      setIsCapturing(true);
      setStatus('Capturing frame...');
      const shot = await cameraRef.current.takePhoto({ flash: 'off' });

      const form = new FormData();
      form.append('file', {
        uri: `file://${shot.path}`,
        name: 'nose.jpg',
        type: 'image/jpeg',
      } as any);

      setStatus('Sending to backend...');
      const res = await fetch(`${BACKEND_URL}/dataset/capture`, {
        method: 'POST',
        body: form,
      });
      const json: CaptureResult & { detail?: string } = await res.json();

      if (!res.ok) {
        const reason = json.detail || json.reason || 'Backend rejected the capture.';
        setLastCapture({
          saved: false,
          capturable: json.capturable ?? false,
          score: json.score,
          overlap: json.overlap,
          quality: json.quality,
          reason,
        });
        setStatus(`Not ready: ${reason}`);
        Alert.alert('Not ready', reason);
        postClientLog('dataset_capture_rejected', {
          reason,
          score: json.score,
          overlap: json.overlap,
          quality: json.quality,
        });
        return;
      }

      setLastCapture(json);

      if (json.saved) {
        setStatus('Saved to dataset');
        setScannerOpen(false);
        Alert.alert('Captured', `Saved sample${json.saved_path ? ` (${json.saved_path})` : ''}.`);
        postClientLog('dataset_capture_saved', {
          score: json.score,
          overlap: json.overlap,
          saved_path: json.saved_path,
        });
      } else {
        const reason = json.reason || 'Nose not centered or quality too low yet.';
        setStatus(`Not ready: ${reason}`);
        Alert.alert('Not ready', reason);
        postClientLog('dataset_capture_rejected', {
          reason,
          score: json.score,
          overlap: json.overlap,
          quality: json.quality,
        });
      }
    } catch (error) {
      console.error(error);
      setStatus('Backend error');
      setConnection('down');
      Alert.alert('Capture failed', 'Could not capture the nose.');
      postClientLog('capture_error', { error: String(error) });
    } finally {
      setIsCapturing(false);
    }
  };

  const handleCloseScanner = () => {
    setScannerOpen(false);
    setStatus('Idle');
  };

  useEffect(() => {
    if (!scannerOpen) return;
    setStatus('Scanner open');
    pollBackendLogs();
    pingBackend();
    const id = setInterval(pollBackendLogs, 4000);
    return () => clearInterval(id);
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

        <TouchableOpacity
          style={[styles.button, { marginTop: 10, backgroundColor: '#1e293b' }]}
          onPress={pingBackend}
          activeOpacity={0.9}
        >
          <Ionicons name="radio" size={20} color="#ffffff" style={{ marginRight: 10 }} />
          <Text style={styles.buttonText}>Test backend</Text>
        </TouchableOpacity>

        <View style={styles.statusPanel}>
          <Text style={styles.statusLabel}>Backend</Text>
          <Text style={styles.statusValue}>{BACKEND_URL}</Text>
          <Text style={styles.statusLabel}>Status</Text>
          <Text style={styles.statusValue}>{status}</Text>
          <Text style={styles.statusLabel}>Connectivity</Text>
          <Text style={styles.statusValue}>
            {connection === 'ok' ? 'Online' : connection === 'down' ? 'Offline' : 'Checking...'}
          </Text>
          <Text style={styles.statusLabel}>Last capture</Text>
          <Text style={styles.statusValue}>
            {lastCapture
              ? lastCapture.saved
                ? `Saved (score ${lastCapture.score ?? '-'}, overlap ${lastCapture.overlap ?? '-'})`
                : `Rejected: ${lastCapture.reason ?? 'not aligned or blurry'}`
              : 'None yet'}
          </Text>
          {lastCapture?.quality && (
            <Text style={styles.statusValue}>
              Quality {lastCapture.quality.passed ? 'ok' : 'needs work'} | Lap {lastCapture.quality.laplacian.toFixed(1)} | Ten {lastCapture.quality.tenengrad.toFixed(1)} | C {lastCapture.quality.contrast.toFixed(1)} | B {lastCapture.quality.brightness.toFixed(1)}
            </Text>
          )}
          {lastCapture?.saved_path && <Text style={styles.statusValue}>Path: {lastCapture.saved_path}</Text>}
        </View>
        <View style={styles.logPanel}>
          <Text style={styles.logTitle}>Backend logs</Text>
          <ScrollView style={styles.logList}>
            {backendLogs.map((log, idx) => (
              <Text key={`${log.ts}-${idx}`} style={styles.logLine}>
                [{log.source}] {log.message}
              </Text>
            ))}
            {backendLogs.length === 0 && <Text style={styles.logLine}>No logs yet</Text>}
          </ScrollView>
        </View>
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
});
