// components/SensorDataDisplay.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, Button, ActivityIndicator } from 'react-native';
import { accelerometer, gyroscope, magnetometer, setUpdateIntervalForType, SensorTypes } from 'react-native-sensors';
import Geolocation, { GeolocationResponse } from '@react-native-community/geolocation';
import NetInfo, { NetInfoState } from '@react-native-community/netinfo';
import { Subscription } from 'rxjs';

// Set sensor update interval
setUpdateIntervalForType(SensorTypes.accelerometer, 500);
setUpdateIntervalForType(SensorTypes.gyroscope, 500);
setUpdateIntervalForType(SensorTypes.magnetometer, 500);

export default function SensorDataDisplay() {
  const [accel, setAccel] = useState({ x: 0, y: 0, z: 0 });
  const [gyro, setGyro] = useState({ x: 0, y: 0, z: 0 });
  const [magneto, setMagneto] = useState({ x: 0, y: 0, z: 0 });
  const [location, setLocation] = useState<GeolocationResponse | null>(null);
  const [netInfo, setNetInfo] = useState<NetInfoState | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Function to manually fetch the current location
  const refreshData = useCallback(() => {
    setIsLoading(true);
    Geolocation.getCurrentPosition(
      position => {
        setLocation(position);
        setIsLoading(false);
      },
      error => {
        console.error(error);
        setIsLoading(false);
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 1000 }
    );
  }, []);

  useEffect(() => {
    // Initial fetch
    refreshData();

    const subscriptions: Subscription[] = [];
    subscriptions.push(accelerometer.subscribe(setAccel));
    subscriptions.push(gyroscope.subscribe(setGyro));
    subscriptions.push(magnetometer.subscribe(setMagneto));

    const watchId = Geolocation.watchPosition(
      pos => setLocation(pos),
      err => console.error(err),
      { enableHighAccuracy: true, distanceFilter: 1, interval: 5000 }
    );

    const netInfoUnsubscribe = NetInfo.addEventListener(setNetInfo);

    return () => {
      subscriptions.forEach(sub => sub.unsubscribe());
      Geolocation.clearWatch(watchId);
      netInfoUnsubscribe();
    };
  }, [refreshData]);

  // Safely get signal strength
  const getSignalStrength = () => {
    if (netInfo?.isConnected && netInfo.details && 'strength' in netInfo.details) {
      if (typeof netInfo.details.strength === 'number') {
        return `${netInfo.details.strength}%`;
      }
    }
    return 'N/A';
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {isLoading && <ActivityIndicator size="large" style={{ marginVertical: 20 }} />}
      
      <View style={styles.section}>
        <Text style={styles.title}>Location Coordinates</Text>
        <Text>Latitude: {location?.coords.latitude?.toFixed(6) ?? 'N/A'}</Text>
        <Text>Longitude: {location?.coords.longitude?.toFixed(6) ?? 'N/A'}</Text>
        <Text>Altitude: {location?.coords.altitude?.toFixed(2) ?? 'N/A'} m</Text>
        <Text>Accuracy: {location?.coords.accuracy?.toFixed(2) ?? 'N/A'} m</Text>
        <Text>Speed: {location?.coords.speed?.toFixed(2) ?? 'N/A'} m/s</Text>
        <Text>Heading: {location?.coords.heading?.toFixed(2) ?? 'N/A'}°</Text>
        <Text>Timestamp: {location ? new Date(location.timestamp).toLocaleTimeString() : 'N/A'}</Text>
      </View>
      
      <View style={styles.section}>
        <Text style={styles.title}>Device Sensors</Text>
        <Text>Accelerometer: X: {accel.x.toFixed(3)}, Y: {accel.y.toFixed(3)}, Z: {accel.z.toFixed(3)}</Text>
        <Text>Gyroscope: X: {gyro.x.toFixed(3)}, Y: {gyro.y.toFixed(3)}, Z: {gyro.z.toFixed(3)}</Text>
        <Text>Magnetometer: X: {magneto.x.toFixed(3)}, Y: {magneto.y.toFixed(3)}, Z: {magneto.z.toFixed(3)}</Text>
      </View>
      
      <View style={styles.section}>
        <Text style={styles.title}>Network & Signal Strength</Text>
        <Text>Type: {netInfo?.type ?? 'N/A'}</Text>
        <Text>Connected: {netInfo?.isConnected ? 'Yes' : 'No'}</Text>
        <Text>Signal Strength: {getSignalStrength()}</Text>
      </View>

      <Button title="Refresh Data" onPress={refreshData} />

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: 20 },
  section: {
    backgroundColor: '#f9f9f9',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#eee',
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
  },
});
