package main

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestSearchRange(t *testing.T) {
	cases := []struct {
		res    []int
		nums   []int
		target int
	}{
		{
			nums:   []int{5, 7, 7, 8, 8, 10},
			res:    []int{3, 4},
			target: 8,
		},
		{
			nums:   []int{5, 7, 7, 8, 8, 10},
			res:    []int{-1, -1},
			target: 6,
		},
		{
			nums:   []int{3, 3, 3},
			res:    []int{0, 2},
			target: 3,
		},
	}

	for _, c := range cases {
		c := c
		t.Run(fmt.Sprintf("SearchRange(%v, %v)", c.nums, c.target), func(t *testing.T) {
			r := searchRange(c.nums, c.target)
			require.Equal(t, c.res, r)
		})
	}
}

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
