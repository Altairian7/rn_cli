import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';

export function LoginScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  function onLogin() {
    if (!email.trim() || !password.trim()) {
      Alert.alert('Missing info', 'Please enter email and password.');
      return;
    }
    // TODO: Replace with real auth call
    Alert.alert('Login', `Email: ${email}`);
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View style={styles.card}>
        <Text style={styles.title}>Welcome back</Text>
        <Text style={styles.subtitle}>Sign in to continue</Text>

        <View style={styles.field}>
          <Text style={styles.label}>Email</Text>
          <TextInput
            placeholder="name@example.com"
            placeholderTextColor="#9ca3af"
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
            value={email}
            onChangeText={setEmail}
            style={styles.input}
          />
        </View>

        <View style={styles.field}>
          <Text style={styles.label}>Password</Text>
          <TextInput
            placeholder="••••••••"
            placeholderTextColor="#9ca3af"
            secureTextEntry
            value={password}
            onChangeText={setPassword}
            style={styles.input}
          />
        </View>

        <TouchableOpacity style={styles.button} onPress={onLogin} activeOpacity={0.85}>
          <Text style={styles.buttonText}>Login</Text>
        </TouchableOpacity>

        <Text style={styles.footerText}>
          By continuing, you agree to our terms.
        </Text>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    justifyContent: 'center',
    backgroundColor: '#f3f4f6', // light gray background
  },
  card: {
    borderRadius: 14,
    padding: 20,
    backgroundColor: '#ffffff',
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
    elevation: 5,
  },
  title: { fontSize: 24, fontWeight: '600', color: '#111827' },
  subtitle: { marginTop: 4, color: '#6b7280' },
  field: { marginTop: 16 },
  label: { color: '#374151', marginBottom: 6, fontWeight: '500' },
  input: {
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: Platform.OS === 'ios' ? 12 : 10,
    backgroundColor: '#f9fafb',
    color: '#111827',
    borderWidth: 1,
    borderColor: '#d1d5db',
  },
  button: {
    marginTop: 18,
    borderRadius: 10,
    backgroundColor: '#2563eb', // blue-600
    alignItems: 'center',
    paddingVertical: 12,
  },
  buttonText: { color: '#ffffff', fontWeight: '600', fontSize: 16 },
  footerText: { marginTop: 10, textAlign: 'center', color: '#6b7280' },
});
