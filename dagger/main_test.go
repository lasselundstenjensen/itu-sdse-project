package main

import (
	"os"
	"testing"
)

// file path validation
// environment flags
// image name : our_group_app

// Test Functions

func TestGetEnvironmentMode(t *testing.T) {
	testCases := []struct {
		name         string
		envValue     string
		expectedMode string
	}{
		{
			name:         "Production mode flag set",
			envValue:     "production",
			expectedMode: "production",
		},
		{
			name:         "Empty flag defaults to development",
			envValue:     "",
			expectedMode: "development",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			os.Setenv("APP_ENV", tc.envValue)

			mode := getAppEnvironment()

			if mode != tc.expectedMode {
				t.Errorf("expected mode %s, got %s", tc.expectedMode, mode)
			}

		})
	}
}

func TestIsValidImageName(name string) bool {
	if name == "" {
		return false
	}
	for _, char := range name {
		if char == ' ' {
			return false
		}
	}
	return true
}

func TestCheckFileExists(t *testing.T) {
	testCases := []struct {
		name        string
		path        string
		expectExist bool
	}{
		{
			name:        "Dockerfile should exist",
			path:        "Dockerfile",
			expectExist: true,
		},
		{
			name:        "Fake file should not exist",
			path:        "non_existent_file.txt",
			expectExist: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			exists := checkFileExists(tc.path)

			// non existent file should return false
			if exists != tc.expectExist {
				t.Errorf("expected file existence to be %v, got %v", tc.expectExist, exists)
			}
		})
	}
}
