import java.util.Vector;

public class VectorOperations {
    public static Vector<Integer> addVectors(Vector<Integer> vec1, Vector<Integer> vec2) {
        if (vec1.size() != vec2.size()) {
            throw new IllegalArgumentException("Vectors must have the same size");
        }

        Vector<Integer> result = new Vector<>();
        for (int i = 0; i < vec1.size(); i++) {
            result.add(vec1.get(i) + vec2.get(i));
        }
        return result;
    }

    public static void main(String[] args) {
        Vector<Integer> vec1 = new Vector<>();
        Vector<Integer> vec2 = new Vector<>();

        vec1.add(1); vec1.add(2); vec1.add(3);
        vec2.add(4); vec2.add(5); vec2.add(6);

        System.out.println("Vector 1: " + vec1);
        System.out.println("Vector 2: " + vec2);

        Vector<Integer> sum = addVectors(vec1, vec2);
        System.out.println("Sum of Vectors: " + sum);
    }
}
