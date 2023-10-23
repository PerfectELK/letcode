package main

import (
	"fmt"
	"github.com/PerfectELK/letcode/linked_list"
	"github.com/PerfectELK/letcode/tree"
	"strings"
)

func main() {
	a := []int{1, 2, 2, 3, 4, 4, 3, 4}
	t := tree.SliceToTree(a)
	fmt.Println(tree.PrintTree(t))
	is := maxDepth(t)
	fmt.Println(is)
}

func maxDepth(root *tree.TreeNode) int {
	if root == nil {
		return 0
	}
	return int(checkDeepNodes(root, 1))
}

func checkDeepNodes(current *tree.TreeNode, depth uint16) uint16 {
	if current.Left == nil && current.Right == nil {
		return depth
	}
	var newDepth uint16
	if current.Left != nil {
		newDepth = checkDeepNodes(current.Left, depth+1)
	}

	if current.Right != nil {
		rd := checkDeepNodes(current.Right, depth+1)
		if rd > newDepth {
			newDepth = rd
		}
	}
	return newDepth
}

func isSymmetric(root *tree.TreeNode) bool {
	path := make([]bool, 0, 100)
	return checkNodes(root, root, path)
}

func checkNodes(current *tree.TreeNode, root *tree.TreeNode, path []bool) bool {
	if current == nil {
		return true
	}
	if len(path) != 0 {
		isSym := checkIsNodeSymmetric(current.Val, root, path)
		if !isSym {
			return false
		}
	}

	lCh := checkNodes(current.Left, root, append(path, false))
	if !lCh {
		return false
	}
	rCh := checkNodes(current.Right, root, append(path, true))
	if !rCh {
		return false
	}
	return true
}

func checkIsNodeSymmetric(val int, root *tree.TreeNode, path []bool) bool {
	cur := root
	for _, p := range path {
		// reverse logic
		if p {
			cur = cur.Left
		} else {
			cur = cur.Right
		}
		if cur == nil {
			return false
		}
	}
	return cur.Val == val
}

func addTwoNumbers(l1 *linked_list.ListNode, l2 *linked_list.ListNode) *linked_list.ListNode {
	a1 := make([]int, 0, 100)
	a2 := make([]int, 0, 100)

	for l1 != nil || l2 != nil {
		if l1 != nil {
			a1 = append(a1, l1.Val)
			l1 = l1.Next
		}
		if l2 != nil {
			a2 = append(a2, l2.Val)
			l2 = l2.Next
		}
	}
	a1I := 0
	a2I := 0
	hArr := a1
	hArrI := a1I
	if len(a2) > len(a1) {
		hArr = a2
		hArrI = a2I
	}
	transfer := 0
	for a1I < len(a1) || a2I < len(a2) {
		n1 := 0
		n2 := 0
		if a1I < len(a1) {
			n1 = a1[a1I]
		}
		if a2I < len(a2) {
			n2 = a2[a2I]
		}
		sum := n1 + n2 + transfer
		if sum >= 10 {
			transfer = 1
			sum = sum % 10
		} else {
			transfer = 0
		}
		hArr[hArrI] = sum
		a1I++
		a2I++
		hArrI++
	}
	lr := new(linked_list.ListNode)
	begin := lr
	for i, val := range hArr {
		lr.Val = val
		if i == len(hArr)-1 {
			break
		}
		lr.Next = new(linked_list.ListNode)
		lr = lr.Next
	}
	if transfer > 0 {
		lr.Next = new(linked_list.ListNode)
		lr = lr.Next
		lr.Val = transfer
	}

	return begin
}

func mergeTwoLists(list1 *linked_list.ListNode, list2 *linked_list.ListNode) *linked_list.ListNode {
	if list1 == nil && list2 == nil {
		return nil
	}
	result := list1
	if result == nil || result.Val > list2.Val {
		result = list2
		list2 = list2.Next
	} else {
		list1 = list1.Next
	}

	var begin = result
	for list1 != nil || list2 != nil {
		if list1 != nil && (list2 == nil || list1.Val <= list2.Val) {
			result.Next = list1
			list1 = list1.Next
			result = result.Next
		}
		if list2 != nil && (list1 == nil || list2.Val <= list1.Val) {
			result.Next = list2
			list2 = list2.Next
			result = result.Next
		}
	}

	return begin
}

func reverseList(head *linked_list.ListNode) *linked_list.ListNode {
	if head == nil {
		return nil
	}
	if head.Next == nil {
		return head
	}
	var prevNode *linked_list.ListNode
	prevNode = nil
	current := head
	for {
		if current.Next == nil {
			current.Next = prevNode
			break
		}
		next := current.Next
		current.Next = prevNode
		prevNode = current
		current = next
	}
	return current
}

func addBinary(a string, b string) string {
	var transfer int
	sb := strings.Builder{}
	aI := len(a) - 1
	bI := len(b) - 1
	iMap := make(map[uint8]int)
	iMap[48] = 0
	iMap[49] = 1
	chMap := make(map[int]byte)
	chMap[0] = 48
	chMap[1] = 49
	for {
		if aI < 0 && bI < 0 {
			if transfer == 1 {
				sb.WriteByte(chMap[1])
			}
			break
		}
		aInt := 0
		if aI >= 0 {
			aInt = iMap[a[aI]]
		}
		bInt := 0
		if bI >= 0 {
			bInt = iMap[b[bI]]
		}
		someHas1 := 0
		if aInt == 1 || bInt == 1 {
			someHas1 = 1
		}
		n := aInt ^ bInt
		n = n ^ transfer
		sb.WriteByte(chMap[n])
		if (aInt == 1 && bInt == 1) || (someHas1 == 1 && transfer == 1) {
			transfer = 1
		} else {
			transfer = 0
		}
		aI--
		bI--
	}
	str := sb.String()
	sbr := strings.Builder{}
	for i := len(str) - 1; i >= 0; i-- {
		sbr.WriteByte(str[i])
	}
	return sbr.String()
}

func lengthOfLastWord(s string) int {
	var lastC int
	var firtsLastC int
	for i, ch := range s {
		if ch != ' ' && (i == 0 || s[i-1] == ' ') {
			firtsLastC = i
		}
		if ch != ' ' {
			lastC = i
		}
	}
	return lastC - firtsLastC + 1
}

func twoSum(nums []int, target int) []int {
	m := make(map[int]int)
	for i, num := range nums {
		if val, ok := m[target-num]; ok {
			return []int{i, val}
		}
		m[num] = i
	}
	return nil
}

// ASCII
func isPalindrome(s string) bool {
	sb := strings.Builder{}
	for _, ch := range s {
		if ch >= 91 && ch <= 96 {
			continue
		}
		if (ch >= 65 && ch <= 122) || (ch >= 48 && ch <= 57) {
			if 'A' <= ch && ch <= 'Z' {
				ch += 'a' - 'A'
			}
			sb.WriteByte(byte(ch))
		}
	}
	sf := sb.String()
	if sf == "" {
		return true
	}
	if len(sf) == 1 {
		return true
	}
	for i, ch := range sf {
		i2 := len(sf) - 1 - i
		if i2 < len(sf)/2 {
			break
		}
		if i == i2 {
			break
		}
		if ch != int32(sf[i2]) {
			return false
		}

	}
	return true
}
