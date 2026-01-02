use crate::Solution;

#[test]
fn convert_to_title(){
    struct TestCase {
        column_number: i32,
        result: String,
    }

    let cases = vec![
        TestCase{column_number: 0, result: "".to_string()},
        TestCase{column_number: 1, result: "A".to_string()},
        TestCase{column_number: 26, result: "Z".to_string()},
        TestCase{column_number: 52, result: "AZ".to_string()},
        TestCase{column_number: 28, result: "AB".to_string()},
        TestCase{column_number: 701, result: "ZY".to_string()},
        TestCase{column_number: 703, result: "AAA".to_string()},
    ];

    for test_case in cases {
        let result = Solution::convert_to_title(test_case.column_number);
        assert_eq!(
            test_case.result, result,
            "Input {}", test_case.column_number
        );
    }
}