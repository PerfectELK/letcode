package tree

import "strconv"

func getLevel(root *TreeNode) (res int) {
	res = 1
	queue := make([]*TreeNode, 1, 1024)
	queue[0] = root

	for {
		hasChild := false
		for i := 0; i < len(queue); i++ {
			if queue[i].Left != nil || queue[i].Right != nil {
				hasChild = true
				break
			}
		}

		if !hasChild {
			break
		}
		res++

		fQueue := queue[:len(queue)]
		queue = queue[len(queue):]

		for i := 0; i < len(fQueue); i++ {
			if fQueue[i].Left != nil {
				queue = append(queue, fQueue[i].Left)
			}
			if fQueue[i].Right != nil {
				queue = append(queue, fQueue[i].Right)
			}
		}
	}

	return res
}

func PrintTree(root *TreeNode) [][]string {
	level := getLevel(root)
	size := 1<<uint(level) - 1
	res := make([][]string, level)
	for i := range res {
		res[i] = make([]string, size)
	}

	loc := 1<<uint(level-1) - 1
	getRes(root, 0, loc, res)

	return res
}

func getRes(root *TreeNode, i, j int, res [][]string) {
	if root == nil {
		return
	}

	res[i][j] = strconv.Itoa(root.Val)

	level := len(res)
	if level-i-2 < 0 {
		return
	}

	diff := 1 << uint(level-i-2)

	getRes(root.Left, i+1, j-diff, res)
	getRes(root.Right, i+1, j+diff, res)
}

func SliceToTree(arr []int) *TreeNode {
	res := new(TreeNode)
	res.Val = arr[0]
	depth := 0
	binCounter := 2
	counter := 0
	for _, val := range arr[1:] {
		if counter == binCounter {
			binCounter *= 2
			depth++
			counter = 0
		}
		fillNode(res, val, 0, depth)
		counter++
	}
	return res
}

func fillNode(current *TreeNode, val int, depth int, maxDepth int) bool {
	if depth > maxDepth {
		return false
	}
	if current.Left == nil {
		current.Left = new(TreeNode)
		current.Left.Val = val
		return true
	}
	if current.Right == nil {
		current.Right = new(TreeNode)
		current.Right.Val = val
		return true
	}
	depth = depth + 1
	isFilled := fillNode(current.Left, val, depth, maxDepth)
	if isFilled {
		return true
	}
	return fillNode(current.Right, val, depth, maxDepth)
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}
