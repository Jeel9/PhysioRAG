import java.util.Collections;
import java.util.Vector;

public class VectorOperations {
    public static void sortVector(Vector<Integer> vec) {
        Collections.sort(vec);
    }

    public static void main(String[] args) {
        Vector<Integer> vec = new Vector<>();
        vec.add(5);
        vec.add(2);
        vec.add(9);
        vec.add(1);

        System.out.println("Before Sorting: " + vec);
        sortVector(vec);
        System.out.println("After Sorting: " + vec);
    }
}
