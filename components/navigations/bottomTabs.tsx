// navigation/BottomTabs.tsx
import React from 'react';
import { NavigationContainer, RouteProp } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { LoginScreen } from '../LoginScreen';
import CameraScreen from '../cameraScreen';
import SensorPermissions from '../SensorPermissions';
import SensorDataDisplay from '../SensorDataDisplay';
import ImgPickerScreen from '../ImgPickerScreen';
import GnssLoggerScreen from '../GnssLoggerScreen';

// Define the parameters for each tab
type RootTabParamList = {
  Login: undefined;
  Camera: undefined;
  Image: undefined;
  Permissions: undefined;
  Data: undefined;
  GNSS: undefined;
};

const Tab = createBottomTabNavigator<RootTabParamList>();

export default function BottomTabs() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }: { route: RouteProp<RootTabParamList, keyof RootTabParamList> }) => ({
          headerShown: false,
          tabBarIcon: ({ focused, color, size }: { focused: boolean; color: string; size: number }) => {
            let iconName = '';

            if (route.name === 'Login') {
              iconName = focused ? 'home' : 'home-outline';
            } else if (route.name === 'Camera') {
              iconName = focused ? 'camera' : 'camera-outline';
            } else if (route.name === 'Image') {
              iconName = focused ? 'image' : 'image-outline';
            } else if (route.name === 'Permissions') {
              iconName = focused ? 'shield-checkmark' : 'shield-checkmark-outline';
            } else if (route.name === 'Data') {
              iconName = focused ? 'analytics' : 'analytics-outline';
            } else if (route.name === 'GNSS') {
              iconName = focused ? 'navigate' : 'navigate-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: 'blue',
          tabBarInactiveTintColor: 'gray',
        })}
      >
        <Tab.Screen name="Login" component={LoginScreen} />
        <Tab.Screen name="Camera" component={CameraScreen} />
        <Tab.Screen name="Image" component={ImgPickerScreen} />
        <Tab.Screen name="Permissions" component={SensorPermissions} />
        <Tab.Screen name="Data" component={SensorDataDisplay} />
        <Tab.Screen name="GNSS" component={GnssLoggerScreen} options={{ tabBarLabel: 'GNSS' }} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
