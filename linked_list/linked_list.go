package linked_list

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
