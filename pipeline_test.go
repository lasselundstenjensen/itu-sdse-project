package main

import "testing"

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
