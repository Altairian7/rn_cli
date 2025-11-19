import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  Alert,
  PermissionsAndroid,
  Platform,
  NativeModules,
  NativeEventEmitter,
  Share,
} from 'react-native';

interface RnGnssModule {
  startLogging: (durationSec: number) => Promise<string>;
  stopLogging: () => Promise<string>;
  getLatestLogFile: () => Promise<string>;
}

const RnGnss = NativeModules.RnGnss as RnGnssModule | undefined;
const eventEmitter = RnGnss ? new NativeEventEmitter(NativeModules.RnGnss) : null;

const GnssLoggerScreen: React.FC = () => {
  const [isLogging, setIsLogging] = useState<boolean>(false);
  const [logMessages, setLogMessages] = useState<string[]>([]);
  const [sessionDuration, setSessionDuration] = useState<string>('120');
  const [hasPermissions, setHasPermissions] = useState<boolean>(false);
  const [latestLogFile, setLatestLogFile] = useState<string | null>(null);

  useEffect(() => {
    requestLocationPermissions();

    const logListener = eventEmitter?.addListener('gnss_log', (message: string) => {
      addLog(message);
      if (message.includes('stopped')) {
        setIsLogging(false);
        if (message.includes('File:')) {
          const filePath = message.split('File: ')[1];
          setLatestLogFile(filePath);
        }
      }
    });

    const dataListener = eventEmitter?.addListener('gnss_data', (message: string) => {
      addLog(message);
    });

    return () => {
      logListener?.remove();
      dataListener?.remove();
    };
  }, []);

  const addLog = (message: string): void => {
    const timestamp = new Date().toLocaleTimeString();
    setLogMessages(prev => [...prev, `[${timestamp}] ${message}`]);
  };

  const requestLocationPermissions = async (): Promise<void> => {
    if (Platform.OS !== 'android') {
      setHasPermissions(true);
      addLog('iOS detected — permissions skipped');
      return;
    }

    try {
      const perms = [
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        PermissionsAndroid.PERMISSIONS.ACCESS_BACKGROUND_LOCATION,
      ];

      const granted = await PermissionsAndroid.requestMultiple(perms);

      const fineGranted =
        granted[PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION] ===
        PermissionsAndroid.RESULTS.GRANTED;

      const bgGranted =
        granted[PermissionsAndroid.PERMISSIONS.ACCESS_BACKGROUND_LOCATION] ===
        PermissionsAndroid.RESULTS.GRANTED;

      if (fineGranted && bgGranted) {
        setHasPermissions(true);
        addLog('✓ Location permissions granted');
      } else {
        setHasPermissions(false);
        addLog('✗ Location permissions denied');
        Alert.alert('Permissions Required', 'Location permissions are required for GNSS logging.');
      }
    } catch (err) {
      console.error('Permission request error:', err);
      addLog('✗ Error requesting permissions');
      setHasPermissions(false);
    }
  };

  const handleStartLogging = async (): Promise<void> => {
    if (!hasPermissions) {
      Alert.alert('Error', 'Location permissions not granted');
      addLog('✗ Permissions not granted');
      return;
    }

    if (!RnGnss) {
      addLog('✗ Native module RnGnss not found');
      Alert.alert('Native Module Missing', 'RnGnss native module is not linked.');
      return;
    }

    const durationNum = parseInt(sessionDuration, 10);

    if (isNaN(durationNum) || durationNum <= 0) {
      Alert.alert('Invalid Duration', 'Please enter a valid number greater than 0');
      addLog('✗ Invalid duration entered');
      return;
    }

    try {
      addLog(`⏳ Starting GNSS logging for ${durationNum}s...`);
      const result = await RnGnss.startLogging(durationNum);
      setIsLogging(true);
      addLog(`▶ ${result}`);
    } catch (error: any) {
      console.error('Start logging error:', error);
      addLog(`✗ Failed to start: ${error?.message || 'Unknown error'}`);
      Alert.alert('Error', 'Failed to start GNSS logging');
    }
  };

  const handleStopLogging = async (): Promise<void> => {
    if (!RnGnss) {
      addLog('✗ Native module RnGnss not found');
      return;
    }

    try {
      addLog('⏳ Stopping GNSS logging...');
      const result = await RnGnss.stopLogging();
      setIsLogging(false);
      addLog(`■ ${result}`);
    } catch (error: any) {
      console.error('Stop logging error:', error);
      addLog(`✗ Failed to stop: ${error?.message || 'Unknown error'}`);
      Alert.alert('Error', 'Failed to stop GNSS logging');
    }
  };

  const handleExportJSON = async (): Promise<void> => {
    if (!latestLogFile) {
      Alert.alert('No Data', 'No log file available to export. Please record a session first.');
      addLog('✗ No log file to export');
      return;
    }

    try {
      addLog('📤 Exporting log file...');
      await Share.share({
        title: 'GNSS Log Export',
        message: `GNSS log file location:\n${latestLogFile}`,
        url: `file://${latestLogFile}`,
      });
      addLog('✓ Export dialog opened');
    } catch (error: any) {
      console.error('Export error:', error);
      addLog(`✗ Failed to export: ${error?.message || 'Unknown error'}`);
      Alert.alert('Export Failed', 'Could not export the log file.');
    }
  };

  const clearLogs = (): void => {
    setLogMessages([]);
    addLog('Logs cleared');
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>GNSS Logger</Text>
        <View style={[styles.statusBadge, isLogging ? styles.statusActive : styles.statusIdle]}>
          <Text style={styles.statusText}>{isLogging ? 'Logging…' : 'Idle'}</Text>
        </View>
      </View>

      <View style={styles.inputSection}>
        <Text style={styles.label}>Session Duration (seconds)</Text>
        <TextInput
          style={styles.input}
          value={sessionDuration}
          onChangeText={setSessionDuration}
          keyboardType="numeric"
          placeholder="120"
          editable={!isLogging}
          maxLength={5}
          returnKeyType="done"
        />
        <Text style={styles.hint}>
          Go outside for better GPS signal. Keep the app in foreground while testing.
        </Text>
      </View>

      <View style={styles.buttonRow}>
        <TouchableOpacity
          style={[styles.button, styles.startBtn, isLogging && styles.disabled]}
          onPress={handleStartLogging}
          disabled={isLogging}
        >
          <Text style={styles.buttonText}>START</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.stopBtn, !isLogging && styles.disabled]}
          onPress={handleStopLogging}
          disabled={!isLogging}
        >
          <Text style={styles.buttonText}>STOP</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity
        style={[styles.exportButton, !latestLogFile && styles.disabled]}
        onPress={handleExportJSON}
        disabled={!latestLogFile}
      >
        <Text style={styles.exportButtonText}>📤 EXPORT JSON</Text>
      </TouchableOpacity>

      <View style={styles.logSection}>
        <View style={styles.logHeader}>
          <Text style={styles.logTitle}>Event Log</Text>
          <TouchableOpacity onPress={clearLogs} style={styles.clearBtn}>
            <Text style={styles.clearBtnText}>Clear</Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.logScroll} contentContainerStyle={styles.logContent}>
          {logMessages.length === 0 ? (
            <Text style={styles.empty}>No events yet…</Text>
          ) : (
            logMessages.map((msg, index) => (
              <Text key={index} style={styles.logLine}>
                {msg}
              </Text>
            ))
          )}
        </ScrollView>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5', padding: 16 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  title: { fontSize: 22, fontWeight: '700', color: '#222' },
  statusBadge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16 },
  statusIdle: { backgroundColor: '#C7C7CC' },
  statusActive: { backgroundColor: '#34C759' },
  statusText: { color: '#fff', fontWeight: '600' },
  inputSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  label: { fontSize: 13, fontWeight: '600', color: '#555', marginBottom: 6 },
  input: {
    borderWidth: 1,
    borderColor: '#E5E5EA',
    borderRadius: 8,
    padding: 10,
    fontSize: 16,
    color: '#222',
    backgroundColor: '#FAFAFA',
  },
  hint: { marginTop: 8, fontSize: 12, color: '#6b7280' },
  buttonRow: { flexDirection: 'row', gap: 12, marginBottom: 12 },
  button: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.15,
    shadowRadius: 2,
  },
  startBtn: { backgroundColor: '#34C759' },
  stopBtn: { backgroundColor: '#FF3B30' },
  disabled: { opacity: 0.5 },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  exportButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.15,
    shadowRadius: 2,
  },
  exportButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  logSection: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  logHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  logTitle: { fontSize: 16, fontWeight: '700', color: '#222' },
  clearBtn: { paddingHorizontal: 10, paddingVertical: 6, backgroundColor: '#F3F4F6', borderRadius: 8 },
  clearBtnText: { color: '#374151', fontWeight: '600', fontSize: 12 },
  logScroll: { flex: 1, backgroundColor: '#FAFAFA', borderRadius: 8, padding: 10 },
  logContent: { flexGrow: 1 },
  empty: { textAlign: 'center', color: '#9CA3AF', marginTop: 10, fontStyle: 'italic' },
  logLine: { fontSize: 12, color: '#111827', marginBottom: 6, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
});

export default GnssLoggerScreen;
