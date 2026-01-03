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

#[test]
fn title_to_number(){
    struct TestCase {
        column_title: String,
        result: i32,
    }  
    
    let cases = vec![
        TestCase{column_title: "A".to_string(), result: 1,},
        TestCase{column_title: "Z".to_string(), result: 26,},
        TestCase{column_title: "AZ".to_string(), result: 52,},
        TestCase{column_title: "AAA".to_string(), result: 703,},
    ];
    
    for test_case in cases {
        let result = Solution::title_to_number(test_case.column_title.clone());
        assert_eq!(
            test_case.result, result,
            "Input {}", test_case.column_title
        );
    }
}