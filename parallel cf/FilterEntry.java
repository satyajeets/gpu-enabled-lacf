import java.nio.ByteBuffer;
import edu.rit.gpu.Struct;

class FilterEntry extends Struct {        
    int size;
    byte[] fingerprint;

    FilterEntry() {
        fingerprint = new byte[5];        
        size = 0;
    }
        
    public static long sizeof() {
        return 12;
    }

    @Override
    public void fromStruct(ByteBuffer buff) {
        // TODO Auto-generated method stub
        size = buff.getInt();
//        fingerprint = buff.array();
        fingerprint = buff.array();
    }

    @Override
    public void toStruct(ByteBuffer buff) {
        // TODO Auto-generated method stub
        buff.putInt(size);
        buff.put(fingerprint);
    }
}