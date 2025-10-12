// components/CameraScreen.tsx
import React, { useState, useEffect } from 'react';
import {
  View,
  Button,
  Image,
  StyleSheet,
  Platform,
  PermissionsAndroid,
  Alert,
  Text,
} from 'react-native';
import { launchCamera, type CameraOptions, type ImagePickerResponse } from 'react-native-image-picker';

export default function CameraScreen() {
  const [uri, setUri] = useState<string | undefined>(undefined);
  // State to hold the permission status
  const [hasPermission, setHasPermission] = useState(false);

  // Ask for permission when the component loads
  useEffect(() => {
    async function requestPermissions() {
      if (Platform.OS === 'android') {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.CAMERA,
          {
            title: "Camera Permission",
            message: "App needs access to your camera to take photos.",
            buttonNeutral: "Ask Me Later",
            buttonNegative: "Cancel",
            buttonPositive: "OK"
          }
        );
        setHasPermission(granted === PermissionsAndroid.RESULTS.GRANTED);
      } else {
        // For iOS, permission is handled by the system automatically on first use.
        // We can assume true and let the OS handle the prompt.
        setHasPermission(true);
      }
    }

    requestPermissions();
  }, []);

  async function onCaptureAndUpload() {
    // Check the permission state before proceeding
    if (!hasPermission) {
      Alert.alert('Permission required', 'Camera permission is needed to take a photo.');
      return;
    }

    const options: CameraOptions = { mediaType: 'photo', quality: 0.9, saveToPhotos: true };
    try {
      const result: ImagePickerResponse = await launchCamera(options);
      if (result.didCancel) return;
      const photo = result.assets?.[0];
      if (!photo?.uri) {
        Alert.alert('No image captured');
        return;
      }
      setUri(photo.uri);
      await uploadPhoto(photo.uri);
      Alert.alert('Uploaded', 'Photo uploaded successfully.');
    } catch (e: any) {
      Alert.alert('Error', e?.message ?? 'Unknown error');
    }
  }

  async function uploadPhoto(fileUri: string) {
    const form = new FormData();
    form.append('file', { uri: fileUri, name: 'photo.jpg', type: 'image/jpeg' } as any);
    const resp = await fetch('https://example.com/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'multipart/form-data' },
      body: form,
    });
    if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
  }

  return (
    <View style={styles.container}>
      <Button title="Capture & Upload Photo" onPress={onCaptureAndUpload} />
      {uri ? (
        <View style={styles.previewWrap}>
          <Text style={styles.label}>Preview</Text>
          <Image source={{ uri }} style={styles.preview} />
        </View>
      ) : (
        <Text style={styles.helper}>Tap the button to open the camera.</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, gap: 16, padding: 16, alignItems: 'center' },
  previewWrap: { marginTop: 12, alignItems: 'center' },
  label: { color: '#666', marginBottom: 6 },
  preview: { width: 260, height: 260, borderRadius: 8, backgroundColor: '#eee' },
  helper: { marginTop: 8, color: '#888' },
});
