import React, { useState, useEffect } from 'react';
import { View, Text, Button, Alert, StyleSheet, Platform, ScrollView } from 'react-native';
import { check, request, PERMISSIONS, RESULTS, openSettings } from 'react-native-permissions';

const permissionMap = {
  motion: Platform.OS === 'ios' ? PERMISSIONS.IOS.MOTION : PERMISSIONS.ANDROID.BODY_SENSORS,
  location: Platform.OS === 'ios' ? PERMISSIONS.IOS.LOCATION_WHEN_IN_USE : PERMISSIONS.ANDROID.ACCESS_FINE_LOCATION,
  gnss: Platform.OS === 'ios' ? PERMISSIONS.IOS.LOCATION_ALWAYS : PERMISSIONS.ANDROID.ACCESS_BACKGROUND_LOCATION,
  camera: Platform.OS === 'ios' ? PERMISSIONS.IOS.CAMERA : PERMISSIONS.ANDROID.CAMERA,
};

type PermissionType = keyof typeof permissionMap;

export default function SensorPermissions() {
  const [statuses, setStatuses] = useState<Record<PermissionType, typeof RESULTS[keyof typeof RESULTS] | string>>({
    motion: 'unavailable',
    location: 'unavailable',
    gnss: 'unavailable',
    camera: 'unavailable',
  });

  const checkAllPermissions = async () => {
    const newStatuses: any = {};
    for (const key in permissionMap) {
      const type = key as PermissionType;
      const permission = permissionMap[type];
      const status = await check(permission);
      newStatuses[type] = status;
    }
    setStatuses(newStatuses);
  };

  useEffect(() => {
    checkAllPermissions();
  }, []);

  const requestPermission = async (type: PermissionType) => {
    const permission = permissionMap[type];
    let result = await request(permission);

    if (result === RESULTS.BLOCKED) {
      Alert.alert(
        `${type.charAt(0).toUpperCase() + type.slice(1)} Permission Blocked`,
        `Please enable this permission in your device settings.`,
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Open Settings', onPress: () => openSettings() },
        ]
      );
    }

    setStatuses(prev => ({ ...prev, [type]: result }));
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>App Permissions</Text>
      <Text style={styles.text}>Request the permissions needed for the app to function correctly.</Text>

      <View style={styles.permissionRow}>
        <Text style={styles.permissionText}>Motion sensors</Text>
        <Button title={`Status: ${statuses.motion}`} onPress={() => requestPermission('motion')} />
      </View>
      
      <View style={styles.permissionRow}>
        <Text style={styles.permissionText}>Location</Text>
        <Button title={`Status: ${statuses.location}`} onPress={() => requestPermission('location')} />
      </View>
      
      <View style={styles.permissionRow}>
        <Text style={styles.permissionText}>GNSS / Background Location</Text>
        <Button title={`Status: ${statuses.gnss}`} onPress={() => requestPermission('gnss')} />
      </View>
      
      <View style={styles.permissionRow}>
        <Text style={styles.permissionText}>Camera</Text>
        <Button title={`Status: ${statuses.camera}`} onPress={() => requestPermission('camera')} />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    justifyContent: 'flex-start',
    padding: 20,
    backgroundColor: '#f7f7f7',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 24,
    letterSpacing: 0.5,
  },
  text: {
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 30,
    color: '#666',
    lineHeight: 24,
  },
  permissionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 18,
    paddingHorizontal: 10,
    marginBottom: 12,
    borderRadius: 8,
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  permissionText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#444',
  },
  button: {
    backgroundColor: '#007bff',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: '#007bff',
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    textAlign: 'center',
  },
  footer: {
    marginTop: 20,
    textAlign: 'center',
    color: '#999',
    fontSize: 14,
  },
});
