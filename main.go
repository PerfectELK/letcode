package main

import (
	"fmt"
	"github.com/PerfectELK/letcode/tree"
	"math"
	"strings"
	"unicode"
)

func main() {
	r := combinationSum([]int{2, 3, 6, 7}, 13) // 2 2 3 6, 7 6
	fmt.Println(r)
}

func combinationSum(candidates []int, target int) [][]int {
	var retArr [][]int

	candidateTable := make(map[int][]int)
	for i, candidate := range candidates {
		if candidate > target {
			continue
		}
		if candidate == target {
			retArr = append(retArr, []int{candidate})
			continue
		}
		if target%candidate == 0 {
			nums := target / candidate
			pArr := make([]int, nums)
			fillIntArr(pArr, candidate, nums)
			retArr = append(retArr, pArr)
			continue
		}
		for i2, candidate2 := range candidates {
			if i == i2 {
				continue
			}
			if candidate+candidate2 <= target {
				if v, ok := candidateTable[candidate]; ok {
					candidateTable[candidate] = append(v, candidate2)
				} else {
					candidateTable[candidate] = []int{candidate2}
				}
			}
		}
	}
	fmt.Println(candidateTable)

	return retArr
}

func fillIntArr(arr []int, num int, amount int) {
	for i := 0; i < amount; i++ {
		arr[i] = num
	}
}

//func plusUntilNeedFound(n int, target int) int {
//	for {
//		if n == target {
//			return
//		}
//		n += n
//	}
//}

func removeElement(nums []int, val int) int {
	ret := len(nums)
	fillIndex := -1
	for i, v := range nums {
		if v == val {
			ret--
			if fillIndex == -1 {
				fillIndex = i
			}
		} else if fillIndex != -1 {
			nums[fillIndex] = v
			fillIndex++
		}
	}
	return ret
}

func searchInsert(nums []int, target int) int {
	searchI := len(nums)/2 - 1
	if searchI < 0 {
		searchI = 0
	}
	step := len(nums) / 2 / 2
	if step == 0 {
		step = 1
	}
	prevValue := 0
	for {
		currVal := nums[searchI]
		if currVal == target {
			return searchI
		}

		if currVal < target {
			searchI += step
			if len(nums)-1 < searchI || nums[searchI] == prevValue {
				return searchI
			}
		} else if currVal > target {
			searchI -= step
			if searchI < 0 {
				return 0
			}
			if nums[searchI] == prevValue {
				return searchI + 1
			}
		}

		step /= 2
		if step == 0 {
			step = 1
		}
		prevValue = currVal
	}
}

var PARENTHESES_MAP = map[int32]int32{
	'{': '}',
	'}': '{',
	'[': ']',
	']': '[',
	'(': ')',
	')': '(',
}

func isValid(s string) bool {
	nearestToClose := make([]int32, 0, len(s))
	for _, ch := range s {
		if _, ok := PARENTHESES_MAP[ch]; !ok {
			return false
		}

		if ch == '{' || ch == '(' || ch == '[' {
			nearestToClose = append(nearestToClose, ch)
		}
		if (ch == '}' || ch == ')' || ch == ']') && len(nearestToClose) != 0 {
			if nearestToClose[len(nearestToClose)-1] != PARENTHESES_MAP[ch] {
				return false
			} else {
				nearestToClose = nearestToClose[:len(nearestToClose)-1]
			}
		} else if (ch == '}' || ch == ')' || ch == ']') && len(nearestToClose) == 0 {
			return false
		}
	}
	if len(nearestToClose) != 0 {
		return false
	}
	return true
}

type ListNode struct {
	Next *ListNode
	Val  int
}

func SliceToListNodes(arr []int) *ListNode {
	l := new(ListNode)
	begin := l
	for i, num := range arr {
		l.Val = num
		if i == len(arr)-1 {
			break
		}
		l.Next = new(ListNode)
		l = l.Next
	}
	return begin
}

func mergeKLists(lists []*ListNode) *ListNode {
	endAmount := 0
	retVal := new(ListNode)
	startRetVal := retVal
	if len(lists) == 0 {
		return nil
	}
	var prevVal *ListNode
	for {
		if endAmount == len(lists) {
			break
		}

		smallest := math.MaxInt64
		smallestLinkIndex := -1
		for i := 0; i < len(lists); i++ {
			if lists[i] == nil {
				continue
			}
			if lists[i].Val < smallest {
				smallest = lists[i].Val
				smallestLinkIndex = i
			}
		}
		if smallestLinkIndex == -1 && endAmount == 0 {
			return nil
		}
		if smallestLinkIndex == -1 {
			endAmount++
			continue
		}
		retVal.Val = smallest
		retVal.Next = new(ListNode)
		prevVal = retVal
		retVal = retVal.Next
		if lists[smallestLinkIndex].Next == nil {
			endAmount++
			lists[smallestLinkIndex] = nil
		} else {
			lists[smallestLinkIndex] = lists[smallestLinkIndex].Next
		}
	}
	prevVal.Next = nil
	return startRetVal
}

func search(nums []int, target int) int {
	if len(nums) == 0 {
		return -1
	}
	m := len(nums) / 2
	if m == 0 {
		if nums[0] == target {
			return 0
		}
		return -1
	}

	step := m / 2
	if step == 0 {
		step = 1
	}
	lI, rI := 0, len(nums)-1
	if rI > len(nums)-1 {
		rI -= 1
	}

	for i := 0; i < len(nums); i++ {
		if lI == -1 && rI == -1 {
			return -1
		}
		if lI != -1 && nums[lI] == target {
			return lI
		}
		if lI == len(nums)-1 {
			lI = -1
		}
		if rI != -1 && nums[rI] == target {
			return rI
		}
		if rI == 0 {
			rI = -1
		}

		if lI != -1 {
			if lI-1 >= 0 && nums[lI-1] > nums[lI] {
				lI = lI - 1
			} else if nums[lI] > target {
				lI = lI - step
			} else if nums[lI] < target {
				if nums[lI] > nums[lI+step] {
					lStep := step / 2
					for {
						if nums[lI] > nums[lI+lStep] {
							lStep /= 2
							if lStep == 0 {
								lStep = 1
							}
						} else {
							lI = lI + lStep
							break
						}
						if lStep == 1 {
							break
						}
					}
				} else {
					lI = lI + step
				}
			}
		}

		if rI != -1 {
			if nums[rI] < target {
				rI = rI + step
			} else if nums[rI] > target {
				if nums[rI] < nums[rI-step] {
					rStep := step / 2
					for {
						if nums[rI] > nums[rI-rStep] {
							rStep /= 2
							if rStep == 0 {
								rStep = 1
							}
						} else {
							rI = rI - rStep
							break
						}
						if rStep == 1 {
							break
						}
					}
				} else {
					rI = rI - step
				}
			}
		}

		if lI < 0 || lI > len(nums)-1 {
			lI = -1
		}

		if rI < 0 || rI > len(nums)-1 {
			rI = -1
		}

		step /= 2
		if step == 0 {
			step = 1
		}
	}
	return -1
}

var ROMAN_MAP = map[uint8]int{
	'I': 1,
	'V': 5,
	'X': 10,
	'L': 50,
	'C': 100,
	'D': 500,
	'M': 1000,
}

func romanToInt(s string) int {
	num := 0
	for i := 0; i < len(s); i++ {
		ch1 := s[i]
		var ch2 uint8
		if i+1 < len(s) {
			ch2 = s[i+1]
		}
		if ROMAN_MAP[ch2] > ROMAN_MAP[ch1] {
			num += sumRoman(ROMAN_MAP[ch1], ROMAN_MAP[ch2])
			i++
		} else {
			num += ROMAN_MAP[ch1]
		}
	}
	return num
}

func sumRoman(ch1 int, ch2 int) int {
	if ch1 == 1 {
		if ch2 == 5 || ch2 == 10 {
			return ch2 - ch1
		}
	}

	if ch1 == 10 {
		if ch2 == 50 || ch2 == 100 {
			return ch2 - ch1
		}
	}

	if ch1 == 100 {
		if ch2 == 500 || ch2 == 1000 {
			return ch2 - ch1
		}
	}
	return 0
}

func plusOne(digits []int) []int {
	isTransfer := true
	for i := len(digits) - 1; i >= 0; i-- {
		if !isTransfer {
			return digits
		}
		digit := digits[i]
		if isTransfer {
			digit += 1
			isTransfer = false
		}
		if digit == 10 {
			digit = 0
			isTransfer = true
		}
		digits[i] = digit
	}

	if isTransfer {
		digits = append([]int{1}, digits...)
	}

	return digits
}

func removeDuplicates(nums []int) int {
	if len(nums) == 1 {
		return 1
	}
	containMap := make(map[int]bool)
	insertIndex := -1
	for i, num := range nums {
		if _, ok := containMap[num]; !ok {
			containMap[num] = true
			if insertIndex != -1 {
				nums[insertIndex] = num
				insertIndex++
			}
		} else {
			if insertIndex == -1 {
				insertIndex = i
			}
		}
	}
	if insertIndex == -1 {
		return len(nums)
	}
	return insertIndex
}

func myAtoi(s string) int {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return 0
	}
	num := 0
	for i, ch := range s {
		if i == 0 && (ch == '-' || ch == '+') {
			continue
		}
		if !unicode.IsDigit(ch) {
			break
		}

		num = num*10 + int(ch-'0')
		if num > math.MaxInt32 && s[0] == '-' {
			return math.MinInt32
		} else if num > math.MaxInt32 {
			return math.MaxInt32
		}
	}

	if s[0] == '-' {
		num *= -1
	}

	return num
}

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	arr := mergeSlices(nums1, nums2)
	if len(arr)%2 != 0 {
		return float64(arr[(len(arr) / 2)])
	}

	return float64(arr[(len(arr)/2)]+arr[(len(arr)/2)-1]) / 2
}

func mergeSlices(nums1 []int, nums2 []int) []int {
	if len(nums1) == 0 {
		return nums2
	}
	if len(nums2) == 0 {
		return nums1
	}

	nums3 := make([]int, len(nums1)+len(nums2))

	var i1, i2, i3 int
	isN1Finish, isN2Finish := false, false

	for i3 < len(nums1)+len(nums2) {
		if isN2Finish {
			nums3[i3] = nums1[i1]
			i1++
		} else if isN1Finish {
			nums3[i3] = nums2[i2]
			i2++
		} else if nums2[i2] < nums1[i1] {
			nums3[i3] = nums2[i2]
			i2++
			if len(nums2) == i2 {
				isN2Finish = true
			}
		} else {
			nums3[i3] = nums1[i1]
			i1++
			if len(nums1) == i1 {
				isN1Finish = true
			}
		}
		i3++
	}

	return nums3
}

func reverse(x int) int {
	if x == 0 {
		return x
	}
	m := 1
	if x < 0 {
		x *= -1
		m = -1
	}
	numAmount := int(math.Log10(float64(x)) + 1)
	nums := make([]int, 0, 100)
	for i := numAmount - 1; i >= 0; i-- {
		pow := int(math.Pow(10, float64(i)))
		n := x / pow
		x = x % pow
		nums = append(nums, n)
	}
	res := 0
	for i := len(nums) - 1; i >= 0; i-- {
		res += nums[i] * int(math.Pow(10, float64(i)))
	}
	posRes := res
	if res < 0 {
		posRes *= -1
	}
	if posRes > int(math.Pow(2, 31)-1) {
		return 0
	}
	res *= m
	return res
}

func mySqrt(x int) int {
	if x == 0 {
		return 0
	}
	if x == 1 {
		return 1
	}
	lowest := 0
	i := int(math.Log10(float64(x)))
	for i < x {
		m := i * i
		if m == x {
			return i
		}
		if m < x {
			lowest = i
		}
		if m > x {
			break
		}
		i++
	}
	return lowest
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

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
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
	lr := new(ListNode)
	begin := lr
	for i, val := range hArr {
		lr.Val = val
		if i == len(hArr)-1 {
			break
		}
		lr.Next = new(ListNode)
		lr = lr.Next
	}
	if transfer > 0 {
		lr.Next = new(ListNode)
		lr = lr.Next
		lr.Val = transfer
	}

	return begin
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
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

func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	if head.Next == nil {
		return head
	}
	var prevNode *ListNode
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
