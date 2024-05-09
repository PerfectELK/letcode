package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

func main() {

	//arr := []int{2, 0, 2, 1, 1, 0}
	//arr := []int{1, 1}
	//arr := []int{0, 2, 1}
	//arr := []int{2, 0, 1}
	//arr := []int{1, 2, 0}
	//arr := []int{0, 0, 1}
	arr := []int{1, 2, 2, 2, 2, 0, 0, 0, 1, 1}

	sortColors(arr)

	fmt.Println(arr)
}

func sortColors(nums []int) {
	if len(nums) == 1 {
		return
	}
	if len(nums) == 2 {
		if nums[0] == 2 {
			nums[0], nums[1] = nums[1], nums[0]
		}
		if nums[1] == 0 {
			nums[0], nums[1] = nums[1], nums[0]
		}
		return
	}
	l, r := 0, len(nums)-1

	for i := 0; i < len(nums); i++ {
		num := nums[i]

		if num == 0 {
			for {
				if l > len(nums)-1 || l >= i {
					break
				}
				if nums[l] <= nums[i] {
					l++
				} else {
					nums[l], nums[i] = num, nums[l]
					break
				}
			}
		} else if num == 2 {
			for {
				if r < 0 || r <= i {
					break
				}
				if nums[r] == 2 {
					r--
				} else {
					nums[r], nums[i] = num, nums[r]
					i--
					break
				}
			}
		}
	}
}

func backtrack(res *[][]int, nums []int, permutation []int, used map[int]struct{}) {
	if len(permutation) == len(nums) {
		c := make([]int, len(permutation))
		copy(c, permutation)
		*res = append(*res, c)
		return
	}

	for _, num := range nums {
		if _, ok := used[num]; ok {
			continue
		}
		used[num] = struct{}{}
		permutation = append(permutation, num)

		backtrack(res, nums, permutation, used)

		delete(used, num)
		permutation = permutation[:len(permutation)-1]
	}
}

func permute(nums []int) [][]int {
	var res [][]int
	backtrack(&res, nums, []int{}, map[int]struct{}{})
	return res
}

func numIdenticalPairs(nums []int) int {
	c := 0
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[i] == nums[j] {
				c++
			}
		}
	}
	return c
}

func defangIPaddr(address string) string {
	return strings.Replace(address, ".", "[.]", -1)
}

func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}

	c := 1

	c += countNodes(root.Left)

	c += countNodes(root.Right)

	return c
}

func hammingWeight(n int) int {
	counter := 0

	for n != 0 {
		if n%2 != 0 {
			counter++
		}
		n = n / 2
	}

	return counter
}

func removeElements(head *ListNode, val int) *ListNode {
	if head == nil {
		return nil
	}

	prev, cur := (*ListNode)(nil), head

	for cur != nil {

		if cur.Val == val {
			if prev == nil {
				head = cur.Next
			} else {
				prev.Next = cur.Next
			}
		} else {
			prev = cur
		}

		cur = cur.Next
	}

	return head
}

func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	return findNeedSum(root, 0, &targetSum)
}

func findNeedSum(n *TreeNode, acc int, target *int) bool {
	acc += n.Val
	if acc == *target && n.Left == nil && n.Right == nil {
		return true
	}

	if n.Left != nil {
		res := findNeedSum(n.Left, acc, target)
		if res {
			return res
		}
	}

	if n.Right != nil {
		res := findNeedSum(n.Right, acc, target)
		if res {
			return res
		}
	}

	return false
}

func singleNumber(nums []int) int {
	m := make(map[int]int, len(nums)/2+1)

	for _, v := range nums {
		m[v]++
	}

	for k, v := range m {
		if v == 1 {
			return k
		}
	}

	return 0
}

func generate(numRows int) [][]int {
	res := make([][]int, numRows)
	if numRows >= 1 {
		res[0] = []int{1}
	}
	if numRows >= 2 {
		res[0] = []int{1}
		res[1] = []int{1, 1}
	}

	if numRows <= 2 {
		return res
	}

	for i := 3; i <= numRows; i++ {
		item := make([]int, i)
		item[0] = 1
		for j := 2; j < i; j++ {
			item[j-1] = res[i-2][j-2] + res[i-2][j-1]
		}
		item[i-1] = 1
		res[i-1] = item
	}

	return res
}

func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	addrs := make(map[*ListNode]bool)

	for {
		if head.Next == nil {
			break
		}
		addrs[head] = true
		if _, ok := addrs[head.Next]; ok {
			return true
		}
		head = head.Next
	}

	return false
}

type validIdx struct {
	start int
	end   int
}

func longestValidParentheses(s string) int {
	if len(s) == 0 {
		return 0
	}
	longestValidParMap := map[byte]int{
		'(': 0,
		')': 0,
	}

	opens := make([]int, 0)
	validIdxs := make([]validIdx, 0)
	cur := 0
	for i, ch := range s {
		if ch == '(' {
			longestValidParMap['(']++
			opens = append(opens, i)
		} else if ch == ')' && longestValidParMap['('] != 0 {
			longestValidParMap['(']--
			opens = opens[0 : len(opens)-1]
		} else if ch == ')' && longestValidParMap['('] == 0 {
			cur = 0
			longestValidParMap['('] = 0
			if len(validIdxs) != 0 && validIdxs[len(validIdxs)-1].end == -1 {
				validIdxs[len(validIdxs)-1].end = i - 1
			}
			continue
		}
		if cur == 1 {
			validIdxs = append(validIdxs, validIdx{
				start: i - 1,
				end:   -1,
			})
		}
		cur++
	}

	if len(validIdxs) != 0 {
		for i, item := range validIdxs {
			if item.end == -1 {
				validIdxs[i].end = len(s) - 1
			}
		}
	}

	longest := 0
	for i := 0; i < len(validIdxs); i++ {
		val := validIdxs[i]
		itemVal := val.end - val.start + 1
		addedIdxs := make([]int, 0)
		addedIdxs = append(addedIdxs, val.start)
		for _, idx := range opens {
			if idx == val.start || idx == val.end {
				itemVal--
			} else if idx >= val.start && idx <= val.end {
				addedIdxs = append(addedIdxs, idx)
			}
		}
		addedIdxs = append(addedIdxs, val.end)
		if len(addedIdxs) > 2 {
			for j := 0; j < len(addedIdxs); j++ {
				if j == len(addedIdxs)-1 {
					break
				}
				vIdx := validIdx{
					start: addedIdxs[j],
					end:   addedIdxs[j+1] - 1,
				}
				if vIdx.start == vIdx.end {
					continue
				}
				if j != 0 {
					vIdx.start = addedIdxs[j] + 1
				}
				if j+1 == len(addedIdxs)-1 {
					vIdx.end = addedIdxs[j+1]
				}
				validIdxs = append(validIdxs, vIdx)
			}
			continue
		}
		if longest < itemVal {
			longest = itemVal
		}
	}

	return longest
}

// Это решение не моё, сохранил потому что показалось супер лаконичным (особенно по сравнению с моим)
//func longestValidParentheses(s string) int {
//	stack := make([]int, 0, len(s))
//	stack = append(stack, -1)
//	res := 0
//
//	for index, char := range s {
//		if char == '(' {
//			stack = append(stack, index)
//		} else {
//			stack = stack[:len(stack)-1]
//			if len(stack) == 0 {
//				stack = append(stack, index)
//			}
//			res = max(res, index-stack[len(stack)-1])
//		}
//		fmt.Println(stack)
//		fmt.Println(index)
//	}
//	return res
//}

func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return checkDepth(root, 1)
}

func checkDepth(t *TreeNode, depth int) int {
	if t.Left == nil && t.Right == nil {
		return depth
	}

	lCh, rCh := 0, 0
	if t.Left != nil {
		lCh = checkDepth(t.Left, depth+1)
	}

	if t.Right != nil {
		rCh = checkDepth(t.Right, depth+1)
	}

	if rCh == 0 && lCh == 0 {
		return depth
	}

	if lCh == 0 {
		return rCh
	}

	if rCh == 0 {
		return lCh
	}

	if lCh < rCh {
		return lCh
	}

	return rCh
}

func findSubstring(s string, words []string) []int {
	if len(words) == 0 {
		return nil
	}
	m := make(map[string]int, len(words))
	for _, word := range words {
		if _, ok := m[word]; ok {
			m[word] = m[word] + 1
		} else {
			m[word] = 1
		}
	}
	wLen := len(words[0])

	matchArr := make([]int, 0)
	matchMap := make(map[string]int)
	firstIdx := -1
	for i := 0; i < len(s)+1-(wLen*len(words)); i++ {
		sI := i
		matches := 0
		for j := 0; j < len(words); j++ {
			subs := s[sI : sI+wLen]
			v1, ok := m[subs]
			if !ok {
				break
			}

			if firstIdx == -1 {
				firstIdx = sI
			}
			if v2, ok := matchMap[subs]; ok {
				if v2 == v1 {
					break
				}
				matchMap[subs] = matchMap[subs] + 1
			} else {
				matchMap[subs] = 1
			}

			matches++
			sI += wLen
			if matches == len(words) {
				matchArr = append(matchArr, firstIdx)
			}
		}
		clear(matchMap)
		firstIdx = -1
	}
	return matchArr
}

func maxArea(height []int) int {
	mArea := 0

	leftI, rightI := 0, len(height)-1
	for leftI < rightI {
		area := calcArea(leftI, rightI, height[leftI], height[rightI])
		if area > mArea {
			mArea = area
		}
		if height[leftI] > height[rightI] {
			rightI--
		} else {
			leftI++
		}
	}
	return mArea
}

func calcArea(lI, rI, lVal, rVal int) int {
	if lVal < rVal {
		return (rI - lI) * lVal
	}
	return (rI - lI) * rVal
}

func min(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

var dirCountMap = map[int][]int{
	1: {0, -1}, // left
	2: {-1, 0}, // up
	3: {0, 1},  // right
	4: {1, 0},  // down
}

func numRookCaptures(board [][]byte) int {
	rookRow, rookCol := findRook(&board)
	if rookRow == -1 && rookCol == -1 {
		return 0
	}

	dirCount := 4

	num := 0
	for dirCount != 0 {
		sRow, sCol := rookRow, rookCol

		for {
			sRow += dirCountMap[dirCount][0]
			sCol += dirCountMap[dirCount][1]
			if sRow == -1 || sRow == 8 {
				break
			}
			if sCol == -1 || sCol == 8 {
				break
			}

			if board[sRow][sCol] == 'B' {
				break
			}
			if board[sRow][sCol] == 'p' {
				num++
				break
			}
		}

		dirCount--
	}

	return num
}

func findRook(board *[][]byte) (int, int) {
	for rowNum, row := range *board {
		for colNum, cell := range row {
			if cell == 'R' {
				return rowNum, colNum
			}
		}
	}
	return -1, -1
}

func swapPairs(head *ListNode) *ListNode {
	cur, prev := head, (*ListNode)(nil)

	counter := 1
	for cur != nil {
		if counter%2 == 0 {
			prev.Val, cur.Val = cur.Val, prev.Val
		}

		counter++
		prev = cur
		cur = cur.Next
	}

	return head
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	cur, need := head, (*ListNode)(nil)
	for cur != nil {
		if need != nil {
			need = need.Next
		}
		if n == 0 {
			need = head
		}
		cur = cur.Next
		n--
	}

	if need == nil {
		return head.Next
	}

	need.Next = need.Next.Next

	return head
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	prevVal := head.Val
	prevLink := head

	l := head.Next
	for {
		if l == nil {
			return head
		}
		if prevVal == l.Val && l.Next != nil {
			prevLink.Next = l.Next
		} else if prevVal == l.Val && l.Next == nil {
			prevLink.Next = nil
			return head
		} else {
			prevLink = l
			prevVal = l.Val
		}

		l = l.Next
	}

}

func climbStairs(n int) int {
	if n < 3 {
		return n
	}

	curr := 2
	next := 3
	for i := 3; i < n+1; i++ {
		curr, next = next, curr+next
	}

	return curr
}

func firstMissingPositive(nums []int) int {
	sort.Ints(nums)
	prev := nums[0]
	if prev > 0 && prev-1 > 0 {
		return 1
	}
	for _, n := range nums[1:] {
		if prev < 0 {
			prev = 0
		}
		if n > 0 && n-prev > 1 {
			return prev + 1
		}
		prev = n
	}
	if nums[len(nums)-1] < 0 {
		return 1
	}
	return nums[len(nums)-1] + 1
}

func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	invertNode(root)
	return root
}

func invertNode(
	root *TreeNode,
) {
	if root == nil {
		return
	}

	l := root.Left
	r := root.Right

	root.Left = r
	root.Right = l

	invertNode(root.Left)
	invertNode(root.Right)
}

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

func isSameTree(p *TreeNode, q *TreeNode) bool {
	return isSameNodes(p, q)
}

func isSameNodes(l *TreeNode, r *TreeNode) bool {
	if l == nil && r == nil {
		return true
	} else if l == nil || r == nil {
		return false
	}

	if l.Val != r.Val {
		return false
	}
	lCh := isSameNodes(l.Left, r.Left)
	if !lCh {
		return false
	}
	rCh := isSameNodes(l.Right, r.Right)
	return rCh
}

func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 || len(strs[0]) == 0 {
		return ""
	}
	if len(strs) == 1 {
		return strs[0]
	}
	longest := []byte(strs[0])
	for _, v := range strs[1:] {
		if len(v) < len(longest) {
			longest = longest[:len(v)]
		}
		for i, b := range v {
			if i > len(longest)-1 {
				longest = longest[:i]
				break
			}
			if longest[i] != byte(b) {
				longest = longest[:i]
				break
			}
		}
		if len(longest) == 0 {
			return ""
		}
	}
	return string(longest)
}

func longestPalindrome(s string) string {
	if len(s) < 2 {
		return s
	}

	start, end := 0, 0
	for i := 0; i < len(s); i++ {
		len1 := expandAroundCenter(s, i, i)
		len2 := expandAroundCenter(s, i, i+1)
		lenMax := max(len1, len2)
		if lenMax > end-start {
			start = i - (lenMax-1)/2
			end = i + lenMax/2
		}
	}
	return s[start : end+1]
}

func expandAroundCenter(s string, left, right int) int {
	L, R := left, right
	for L >= 0 && R < len(s) && s[L] == s[R] {
		L--
		R++
	}
	return R - L - 1
}

func reverseNum(num int) int {
	res := 0
	for num > 0 {
		res = (res * 10) + num%10
		num /= 10
	}
	return res
}

func isPalindromeNum(x int) bool {
	if x < 0 {
		return false
	}
	bs := []byte(strconv.Itoa(x))

	for i, end := 0, len(bs)-1; i <= end; i++ {
		if bs[i] != bs[end] {
			return false
		}
		end--
	}

	return true
}

func strStr(haystack string, needle string) int {
	if len(needle) > len(haystack) {
		return -1
	}
	resI := -1
	checkPtr := 0
	for i := 0; i < len(haystack); {
		ch := haystack[i]
		if ch == needle[checkPtr] {
			if checkPtr == 0 {
				resI = i
			}
			checkPtr++
			if checkPtr == len(needle) {
				return resI
			}
		} else {
			if resI != -1 {
				i = resI
			}
			checkPtr = 0
			resI = -1
		}
		i++
	}
	return -1
}

func merge(nums1 []int, m int, nums2 []int, n int) {
	if n == 0 {
		return
	}
	if m == 0 {
		copy(nums1, nums2)
		return
	}
	i1, i2 := 0, 0
	tmp := make([]int, 0, len(nums1))
	for i1 < len(nums1)-n || i2 < len(nums2) {
		if i1 >= len(nums1)-n {
			tmp = append(tmp, nums2[i2])
			i2++
			continue
		}
		if i2 >= len(nums2) {
			tmp = append(tmp, nums1[i1])
			i1++
			continue
		}
		if nums2[i2] >= nums1[i1] {
			tmp = append(tmp, nums1[i1])
			i1++
		} else {
			tmp = append(tmp, nums2[i2])
			i2++
		}
	}
	copy(nums1, tmp)
}

func maxSubArray(nums []int) int {
	curSum, maxSum := nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		curSum = max(curSum+nums[i], nums[i])
		maxSum = max(curSum, maxSum)
	}
	return maxSum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func reverseByteSlice(arr []byte) {
	end := len(arr) - 1
	for start := 0; start < end; start++ {
		tmp := arr[start]
		arr[start] = arr[end]
		arr[end] = tmp
		end--
	}
}

func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}

	result := ""
	n1I := len(num1) - 1
	transfer := 0
	for ; n1I >= 0; n1I-- {
		transferSB := strings.Builder{}
		n1 := int(num1[n1I] - '0')

		n2I := len(num2) - 1
		var layerArr []byte
		for ; n2I >= 0; n2I-- {
			n2 := int(num2[n2I] - '0')

			n3 := (n2 * n1) + transfer
			if n3 >= 10 {
				transfer = n3 / 10
				n3 = n3 % 10
			} else {
				transfer = 0
			}
			layerArr = append(layerArr, byte(n3+'0'))
		}
		if transfer != 0 {
			layerArr = append(layerArr, byte(transfer+'0'))
			transfer = 0
		}
		reverseByteSlice(layerArr)
		transferSB.Write(layerArr)
		for i := len(num1) - 1; i > n1I; i-- {
			transferSB.WriteByte('0')
		}
		result = addStrings(result, transferSB.String())
	}

	if transfer != 0 {
		result = fmt.Sprintf("%d%s", transfer, result)
	}

	return result
}

func addStrings(num1 string, num2 string) string {
	str := make([]byte, 0, len(num1)+len(num2))

	n1I := len(num1) - 1
	n2I := len(num2) - 1

	transfer := 0
	for n1I >= 0 || n2I >= 0 {
		n1, n2 := 0, 0
		if n1I >= 0 {
			n1 = int(num1[n1I] - '0')
		}
		if n2I >= 0 {
			n2 = int(num2[n2I] - '0')
		}

		n := n1 + n2 + transfer
		if n >= 10 {
			transfer = n / 10
			str = append(str, byte(n%10+'0'))
		} else {
			str = append(str, byte(n+'0'))
			transfer = 0
		}

		n1I--
		n2I--
	}

	if transfer != 0 {
		str = append(str, byte(transfer+'0'))
	}
	reverseByteSlice(str)

	sb := strings.Builder{}
	sb.Write(str)
	return sb.String()
}

func combinationSum(candidates []int, target int) [][]int {
	var retArr [][]int
	if len(candidates) == 0 {
		return retArr
	}
	combine(&retArr, candidates, []int{}, target, 0, 0)
	return retArr
}

func combine(retArr *[][]int, candidates []int, tmp []int, target int, sum int, idx int) {
	if sum > target || idx > len(candidates) {
		return
	}

	if sum == target {
		nTmp := make([]int, len(tmp))
		copy(nTmp, tmp)
		*retArr = append(*retArr, nTmp)
	}

	for ; idx < len(candidates); idx++ {
		tmp = append(tmp, candidates[idx])
		combine(retArr, candidates, tmp, target, candidates[idx]+sum, idx)
		tmp = tmp[:len(tmp)-1]
	}
}

func sumArrays(arr1 []int, arr2 []int) int {
	sum := 0
	for _, val := range arr1 {
		sum += val
	}
	for _, val := range arr2 {
		sum += val
	}
	return sum
}

func fillIntArr(arr []int, num int, amount int) {
	for i := 0; i < amount; i++ {
		arr[i] = num
	}
}

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

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return int(checkDeepNodes(root, 1))
}

func checkDeepNodes(current *TreeNode, depth uint16) uint16 {
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

func isSymmetric(root *TreeNode) bool {
	path := make([]bool, 0, 100)
	return checkNodes(root, root, path)
}

func checkNodes(current *TreeNode, root *TreeNode, path []bool) bool {
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

func checkIsNodeSymmetric(val int, root *TreeNode, path []bool) bool {
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
