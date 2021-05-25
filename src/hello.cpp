#include <iostream>

using namespace std;



void make_index(int w, int h, int np){
    int count = w * h / np;
    int rem = w * h % np;
    int mv_sum = 0;
    for(int i=0; i<np; i++){
        if (i==np-1 and rem != 0) {
            count += rem;
        }
        int s_r = mv_sum / w, s_c = mv_sum % w; mv_sum += count;
        int e_r = mv_sum / w, e_c = mv_sum % w;
        int cnt = 0;
        while(s_r * w + s_c < e_r * w + e_c){
            cnt+=1;
            if((++s_c)==w) {s_c=0; s_r++;}
        }
        cout << cnt << endl;
    }
}

int main(void){
    int w, h, np;
    w = 8, h = 8, np = 7;
    make_index(w, h, np);
    
    return 0;
}


