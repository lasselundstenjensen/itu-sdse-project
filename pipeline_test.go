package main

import (
	"os"
	"path/filepath"
	"testing"
)

// unit test for the ImageRef function which constructs the reference under which an image will be published from the owner and the commit sha
func TestImageRef(t *testing.T) {
	testCases := []struct {
		name    string
		owner   string
		sha     string
		want    string
		wantErr bool
	}{
		{"lowercase owner", "niels04", "abc123", "ghcr.io/niels04/itu-sdse-project:abc123", false},
		{"uppercase owner", "Niels04", "abc123", "ghcr.io/niels04/itu-sdse-project:abc123", false},
		{"empty sha", "niels04", "", "", true},
		{"empty owner", "", "abc123", "", true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := imageRef(tc.owner, tc.sha)
			// raise fatal error if the function returns an error but wasn't supposed to:
			if (err != nil) != tc.wantErr {
				t.Fatalf("error = %v, wantErr = %v", err, tc.wantErr)
			}
			// raise error of the function doesn't return the expected value:
			if out != tc.want && !tc.wantErr {
				t.Errorf("expected value: %s, got: %s, for name: '%s'", tc.want, out, tc.name)
			}
		})
	}
}

// unit test for the checkArtifact function which is supposed to check whether a given directory contains the MLmodel that is necessary for MLflow to load a model
func TestCheckArtifact(t *testing.T) {
	complete := t.TempDir() // temp folders get deleted automatically when the test exits

	// write a dummy MLmodel file to the "complete" dir
	if err := os.WriteFile(filepath.Join(complete, "MLmodel"), []byte("flavors: {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// make an empty temp dir -> function should fail on this!
	empty := t.TempDir()

	tests := []struct {
		name    string
		dir     string
		wantErr bool
	}{
		{"complete model", complete, false},
		{"no MLmodel file", empty, true},
		{"folder missing", filepath.Join(empty, "missing"), true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := checkArtifact(tc.dir)
			if (err != nil) != tc.wantErr {
				t.Errorf("error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}
