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
            placeholderTextColor="#9aa0a6"
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
            placeholderTextColor="#9aa0a6"
            secureTextEntry
            value={password}
            onChangeText={setPassword}
            style={styles.input}
          />
        </View>

        <TouchableOpacity style={styles.button} onPress={onLogin} activeOpacity={0.8}>
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
  container: { flex: 1, padding: 16, justifyContent: 'center', backgroundColor: '#0b1324' },
  card: {
    borderRadius: 14,
    padding: 20,
    backgroundColor: '#111a33',
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 },
    elevation: 6,
  },
  title: { fontSize: 24, fontWeight: '600', color: '#e6e9ef' },
  subtitle: { marginTop: 4, color: '#aab2c8' },
  field: { marginTop: 16 },
  label: { color: '#c7cede', marginBottom: 6 },
  input: {
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: Platform.OS === 'ios' ? 12 : 10,
    backgroundColor: '#0e1730',
    color: '#e6e9ef',
    borderWidth: 1,
    borderColor: '#263357',
  },
  button: {
    marginTop: 18,
    borderRadius: 10,
    backgroundColor: '#3b82f6',
    alignItems: 'center',
    paddingVertical: 12,
  },
  buttonText: { color: '#fff', fontWeight: '600' },
  footerText: { marginTop: 10, textAlign: 'center', color: '#93a0bd' },
});
