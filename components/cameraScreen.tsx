// components/CameraScreen.tsx
import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, Text, TouchableOpacity, Image, Alert } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, PhotoFile } from 'react-native-vision-camera';
import Ionicons from 'react-native-vector-icons/Ionicons';

export default function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const camera = useRef<Camera>(null);
  
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [photo, setPhoto] = useState<PhotoFile | null>(null);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const onCapturePhoto = async () => {
    if (camera.current) {
      try {
        const capturedPhoto = await camera.current.takePhoto({ flash: 'off' });
        setPhoto(capturedPhoto);
        setIsCameraOpen(false); // Close camera to show preview
      } catch (e) {
        console.error('Failed to take photo:', e);
        Alert.alert('Error', 'Failed to take photo.');
      }
    }
  };
  
  const onRetakePhoto = () => {
    setPhoto(null);
    setIsCameraOpen(true); // Re-open camera
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
      setPhoto(null); // Reset after upload
    } catch (e) {
      console.error(e);
      Alert.alert('Upload Failed', 'Failed to upload photo.');
    }
  };

  // --- Render logic based on state ---

  // 1. Handle Permissions
  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.messageText}>Camera permission is required.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // 2. Handle no camera device
  if (device == null) {
    return (
      <View style={styles.container}>
        <Text style={styles.messageText}>No camera device found.</Text>
      </View>
    );
  }

  // 3. Show Photo Preview
  if (photo) {
    return (
      <View style={styles.container}>
        <Image source={{ uri: `file://${photo.path}` }} style={StyleSheet.absoluteFill} />
        <View style={styles.previewControls}>
          <TouchableOpacity style={styles.controlButton} onPress={onRetakePhoto}>
            <Ionicons name="close-circle" size={60} color="white" />
            <Text style={styles.controlText}>Retake</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.controlButton} onPress={onUploadPhoto}>
            <Ionicons name="checkmark-circle" size={60} color="white" />
            <Text style={styles.controlText}>Upload</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // 4. Show Live Camera View
  if (isCameraOpen) {
    return (
      <View style={styles.container}>
        <Camera
          ref={camera}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
          photoQualityBalance="speed"
        />
        <View style={styles.captureButtonContainer}>
          <TouchableOpacity style={styles.captureButton} onPress={onCapturePhoto} />
        </View>
        <TouchableOpacity style={styles.closeButton} onPress={() => setIsCameraOpen(false)}>
          <Ionicons name="close" size={35} color="white" />
        </TouchableOpacity>
      </View>
    );
  }

  // 5. Show Initial "Open Camera" Button
  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.button} onPress={() => setIsCameraOpen(true)}>
        <Ionicons name="camera-outline" size={24} color="white" style={{ marginRight: 10 }}/>
        <Text style={styles.buttonText}>Open Camera</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1c1c1e',
  },
  messageText: {
    color: 'white',
    fontSize: 18,
    marginBottom: 20,
    textAlign: 'center'
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 30,
    flexDirection: 'row',
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  captureButtonContainer: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
  },
  captureButton: {
    width: 75,
    height: 75,
    borderRadius: 40,
    backgroundColor: 'white',
    borderWidth: 5,
    borderColor: 'rgba(0,0,0,0.2)',
  },
  closeButton: {
    position: 'absolute',
    top: 60,
    left: 30,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 20,
    padding: 5,
  },
  previewControls: {
    position: 'absolute',
    bottom: 50,
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
  },
  controlButton: {
    alignItems: 'center',
  },
  controlText: {
    color: 'white',
    fontSize: 16,
    marginTop: 5,
  },
});
