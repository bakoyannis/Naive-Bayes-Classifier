import csv

def transpose_array(array: list) -> list:
    return [[array[j][i] for j in range(len(array))] for i in range(len(array[0]))]

def calc_mean(array: list) -> float:
    arr_len = len(array)
    result = 0
    for value in array:
        result += value
    return result / arr_len

def calc_var(array: list) -> float:
    mean = calc_mean(array)
    
    deviations = []
    for value in array:
        deviations.append(value - mean)
    
    squared_deviations = []
    for value in deviations:
        squared_deviations.append(value ** 2)
    
    sum_of_squared_deviations = 0
    for value in squared_deviations:
        sum_of_squared_deviations += value
    
    return sum_of_squared_deviations / len(array)

class csv_reader:
    @staticmethod
    def read_csv(f_location: str) -> list:
        """Load CSV data from a file."""
        with open(f_location, 'r', encoding='utf-8') as _file:
            lines = csv.reader(_file)
            data = list(lines)
        _file.close()
        return data
    
    @staticmethod
    def determineDataTypes(categories: list) -> list:
        """Determine the data per category."""
        categoryType = []
        i = 1
        while i < len(categories):
            if categories[i].strip() == "C":
                categoryType.append("C")
            else:
                categoryType.append("D")
            i += 2
        return categoryType

    def __init__(self, f_location: str) -> None:
        data = self.read_csv(f_location)
        self.categoryType = self.determineDataTypes(data[0])
        
        # Seperate labels and features
        self.labels = []
        self.features = []
        for row in data[1:]:
            p_row = []
            for value in row[:-1]:
                    p_row.append(value.strip())
            self.features.append(p_row)
            self.labels.append(row[-1])

if __name__ == "__main__":
    data_obj = csv_reader("datasets/iris.csv")
    print(data_obj.labels)