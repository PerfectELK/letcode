package main

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestMaxPoints(t *testing.T) {
	cases := []struct {
		points [][]int
		result int
	}{
		{
			points: [][]int{
				{1, 1}, {2, 2}, {3, 3},
			},
			result: 3,
		},
		{
			points: [][]int{
				{1, 1}, {3, 2}, {5, 3}, {4, 1}, {2, 3}, {1, 4},
			},
			result: 4,
		},
		{
			points: [][]int{
				{0, 0},
			},
			result: 1,
		},
	}

	for _, c := range cases {
		c := c
		t.Run(fmt.Sprintf("MaxPoints(%v)", c.points), func(t *testing.T) {
			r := maxPoints(c.points)
			require.Equal(t, c.result, r)
		})
	}
}

func TestCandy(t *testing.T) {
	cases := []struct {
		ratings []int
		result  int
	}{
		{
			ratings: []int{1, 0, 2},
			result:  5,
		},
		{
			ratings: []int{1, 2, 2},
			result:  4,
		},
		{
			ratings: []int{1, 3, 2, 2, 1},
			result:  7,
		},
		{
			ratings: []int{1, 2, 87, 87, 87, 2, 1},
			result:  13,
		},
	}

	for _, c := range cases {
		c := c
		t.Run(fmt.Sprintf("Candy(%v)", c.ratings), func(t *testing.T) {
			r := Candy(c.ratings)
			require.Equal(t, c.result, r)
		})
	}
}

func TestIsNumber(t *testing.T) {
	cases := []struct {
		res bool
		num string
	}{
		{
			num: "0",
			res: true,
		},
		{
			num: "e",
			res: false,
		},
		{
			num: ".",
			res: false,
		},
		{
			num: "0089",
			res: true,
		},
		{
			num: "2e10",
			res: true,
		},
		{
			num: "1e",
			res: false,
		},
		{
			num: "inf",
			res: false,
		},
		{
			num: "-8115e957",
			res: true,
		},
	}

	for _, c := range cases {
		c := c
		t.Run(fmt.Sprintf("IsNumber(%v)", c.num), func(t *testing.T) {
			r := isNumber(c.num)
			require.Equal(t, c.res, r)
		})
	}
}

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
