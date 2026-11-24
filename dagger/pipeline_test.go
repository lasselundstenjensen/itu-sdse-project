package main

import "testing"

// table driven test, each row is one case: a name, the input, the expected output, and whether an error is expected
func TestParseStage(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    string
		wantErr bool
	}{
		{"test stage", []string{"pipeline", "test"}, "test", false},
		{"train stage", []string{"pipeline", "train"}, "train", false},
		{"no stage given", []string{"pipeline"}, "", true},
		{"unknown stage", []string{"pipeline", "deploy"}, "", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseStage(tc.args)
			if (err != nil) != tc.wantErr {
				t.Fatalf("error = %v, wantErr %v", err, tc.wantErr)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}
