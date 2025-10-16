// components/CameraScreen.tsx
import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, Text, TouchableOpacity, Image, Alert } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, PhotoFile } from 'react-native-vision-camera';
import Ionicons from 'react-native-vector-icons/Ionicons';

export default function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const camera = useRef<Camera>(null);
  
  const [isActive, setIsActive] = useState(true);
  const [photo, setPhoto] = useState<PhotoFile | null>(null);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const onCapturePhoto = async () => {
    if (camera.current) {
      try {
        // Corrected: Removed unsupported options
        const capturedPhoto = await camera.current.takePhoto({
          flash: 'off',
        });
        setPhoto(capturedPhoto);
        setIsActive(false); // Deactivate camera to show preview
      } catch (e) {
        console.error('Failed to take photo:', e);
        Alert.alert('Error', 'Failed to take photo.');
      }
    }
  };
  
  const onRetakePhoto = () => {
    setPhoto(null);
    setIsActive(true); // Reactivate camera
  };

  const onUploadPhoto = async () => {
    if (!photo) return;

    const form = new FormData();
    form.append('file', {
      uri: `file://${photo.path}`,
      name: 'photo.jpg',
      type: 'image/jpeg',
    });

    try {
      const resp = await fetch('https://example.com/upload', {
        method: 'POST',
        body: form,
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
      Alert.alert('Uploaded', 'Photo uploaded successfully!');
    } catch (e) {
      console.error(e);
      Alert.alert('Upload Failed', 'Failed to upload photo.');
    }
  };

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>Camera permission is required.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (device == null) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>No camera device found.</Text>
      </View>
    );
  }

  if (photo) {
    return (
      <View style={styles.container}>
        <Image source={{ uri: `file://${photo.path}` }} style={StyleSheet.absoluteFill} />
        <View style={styles.previewControls}>
          <TouchableOpacity style={styles.controlButton} onPress={onRetakePhoto}>
            <Ionicons name="close-circle" size={50} color="white" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.controlButton} onPress={onUploadPhoto}>
            <Ionicons name="checkmark-circle" size={50} color="white" />
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        photo={true}
        // Use this prop on the Camera component for quality/speed trade-off
        photoQualityBalance="speed" 
      />
      <View style={styles.captureButtonContainer}>
        <TouchableOpacity style={styles.captureButton} onPress={onCapturePhoto} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'black',
  },
  permissionText: {
    color: 'white',
    fontSize: 18,
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#1E90FF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
  captureButtonContainer: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'white',
    borderWidth: 5,
    borderColor: 'rgba(255,255,255,0.5)',
  },
  previewControls: {
    position: 'absolute',
    bottom: 50,
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
  },
  controlButton: {
    padding: 10,
  },
});
