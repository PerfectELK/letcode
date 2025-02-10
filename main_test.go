package main

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestCanJump(t *testing.T) {
	cases := []struct {
		res  bool
		data []int
	}{
		{
			res:  true,
			data: []int{2, 3, 1, 1, 4},
		},
		{
			res:  false,
			data: []int{3, 2, 1, 0, 4},
		},
		{
			res:  false,
			data: []int{0, 5, 6, 7, 8},
		},
		{
			res:  true,
			data: []int{2, 0, 0},
		},
	}

	for _, c := range cases {
		c := c
		t.Run(fmt.Sprintf("canJump(%v)", c.data), func(t *testing.T) {
			r := canJump(c.data)
			require.Equal(t, c.res, r)
		})
	}

}
