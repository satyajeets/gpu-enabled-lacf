import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;

public class GenerateRandom {
    public static void main(String args[]) {
        Random r = new Random();
        String entry = "";
        int one=0, two=0, three=0, four=0, mask=0;
        BufferedWriter br;
        try {
            br = new BufferedWriter(new FileWriter(new File("prefix4.txt")));    
            for ( int i = 0 ; i < 23000 ; i++ ) {
                one = r.nextInt(256);
                two = r.nextInt(256);
                three = r.nextInt(256);
                four = r.nextInt(256);
                mask = r.nextInt(33);
                entry = one + "." + two + "." + three + "." + four + "/" + mask + "\n";
                br.write(entry);
                entry = "";
            }
            br.flush();
    
            //generate lookup file
            br = new BufferedWriter(new FileWriter(new File("lookup4.txt")));
            for ( int i = 0 ; i < 1000 ; i++ ) {
                one = r.nextInt(256);
                two = r.nextInt(256);
                three = r.nextInt(256);
                four = r.nextInt(256);
                entry = one + "." + two + "." + three + "." + four + "\n";
                br.write(entry);
                entry = "";
            }            
            
            br.close();
        } catch ( Exception e ) {
            e.printStackTrace();
        }
    }
}
