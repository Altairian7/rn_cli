// navigation/BottomTabs.tsx
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { StyleSheet } from 'react-native';

import { LoginScreen } from '../LoginScreen';
import CameraScreen from '../cameraScreen';
import SensorPermissions from '../SensorPermissions';

type RootTabParamList = {
  Login: undefined;
  Camera: undefined;
  Permissions: undefined;
};

const Tab = createBottomTabNavigator<RootTabParamList>();

export default function BottomTabs() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName: string;

            switch (route.name) {
              case 'Login':
                iconName = focused ? 'log-in' : 'log-in-outline';
                break;
              case 'Camera':
                iconName = focused ? 'camera' : 'camera-outline';
                break;
              case 'Permissions':
                iconName = focused ? 'shield-checkmark' : 'shield-checkmark-outline';
                break;
              default:
                iconName = 'help-circle-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#1e90ff',
          tabBarInactiveTintColor: 'gray',
          tabBarLabelStyle: styles.tabBarLabel,
          tabBarStyle: styles.tabBar,
          tabBarItemStyle: styles.tabBarItem,
          headerShown: false, // Optional: hide headers for a cleaner tab experience
        })}
      >
        <Tab.Screen name="Login" component={LoginScreen} />
        <Tab.Screen name="Camera" component={CameraScreen} />
        <Tab.Screen name="Permissions" component={SensorPermissions} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#ddd',
    paddingBottom: 10,
    paddingTop: 5,
    height: 65,
  },
  tabBarLabel: {
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 5,
  },
  tabBarItem: {
    paddingHorizontal: 10,
  },
});
