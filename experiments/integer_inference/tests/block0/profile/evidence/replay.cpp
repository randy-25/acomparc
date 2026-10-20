#include <bits/stdc++.h>
using namespace std;
using U=uint64_t; using S=int64_t;
constexpr U BASE=0x80000000ULL;
struct I {U pc;int len,op,a,b,c;S imm;int fn;};
struct Record {int fn,stage;U start,count,ret;array<U,8> args;vector<U>hist;};
vector<unsigned char> memory;
U loadmem(U addr,int bytes,bool sign=false){if(addr<BASE||addr+bytes>BASE+memory.size())throw runtime_error("memory read "+to_string(addr));U v=0;memcpy(&v,&memory[addr-BASE],bytes);if(sign){if(bytes==1)return(S)(int8_t)v;if(bytes==4)return(S)(int32_t)v;}return v;}
void savemem(U addr,U v,int bytes){if(addr<BASE||addr+bytes>BASE+memory.size())throw runtime_error("memory write "+to_string(addr));memcpy(&memory[addr-BASE],&v,bytes);}
int main(){
 ifstream bin("analysis_work/memory.bin",ios::binary);memory.assign(istreambuf_iterator<char>(bin),{});
 ifstream in("analysis_work/program.txt");int no,nf,ni;in>>no>>nf>>ni;vector<string>ops(no),names(nf);vector<U>starts(nf);for(auto&s:ops)in>>s;for(int i=0;i<nf;i++)in>>starts[i]>>names[i];vector<I>code(ni);vector<int>lookup(0x1226/2+1,-1),entry(lookup.size(),-1);for(int i=0;i<nf;i++)entry[(starts[i]-BASE)/2]=i;
 for(int k=0;k<ni;k++){I&i=code[k];in>>i.pc>>i.len>>i.op>>i.a>>i.b>>i.c>>i.imm>>i.fn;lookup[(i.pc-BASE)/2]=k;}
 enum O {LI,LUI,AUIPC,MV,SEXT,ZEXT,LB,LW,LD,SB,SW,SD,ADDI,ADDIW,SLLI,SRLI,SRAI,ANDI,ADD,ADDW,SUB,SUBW,MUL,MULW,SLL,SRA,AND,SLT,DIV,DIVU,BEQ,BNE,BGE,BGEU,BLTU,BEQZ,BNEZ,BGEZ,JAL,J,RET};
 vector<string>ordered={"li","lui","auipc","mv","sext.w","zext.b","lb","lw","ld","sb","sw","sd","addi","addiw","slli","srli","srai","andi","add","addw","sub","subw","mul","mulw","sll","sra","and","slt","div","divu","beq","bne","bge","bgeu","bltu","beqz","bnez","bgez","jal","j","ret"};vector<int>dispatch(no);for(int i=0;i<no;i++){auto p=find(ordered.begin(),ordered.end(),ops[i]);if(p==ordered.end())throw runtime_error(ops[i]);dispatch[i]=p-ordered.begin();}
 U workspace=BASE+0xe00000,sp=BASE+0xf00000;
 for(int n=0;n<18;n++)savemem(workspace+n*8,loadmem(BASE+0x1e88+n*8,8),8);
 U output=loadmem(workspace+24,8);U t0=0x801877d8ULL,golden=0x800487d8ULL;
 for(int cas=0;cas<8;cas++){
  U r[32]={};r[2]=sp;r[10]=t0+cas*4096;r[11]=output;r[12]=workspace;
  U pc=0x800011b8,n=0;vector<U>counts(ni),taken(ni);vector<vector<U>>stagecounts(7,vector<U>(ni));vector<Record>rec;vector<int>stack;int stage=-1;
  vector<string>stage_names={"block0_ln1_forward","block0_qkv_forward","block0_attention_forward","block0_out_residual_forward","block0_ln2_forward","block0_mlp_gelu_forward","block0_mlp_out_residual_forward"};
  while(pc){
   if(pc<BASE||pc>=BASE+0x1226||lookup[(pc-BASE)/2]<0)throw runtime_error("bad pc "+to_string(pc));
   int ix=lookup[(pc-BASE)/2];const I&i=code[ix];int en=entry[(pc-BASE)/2];
   if(en>=0){auto it=find(stage_names.begin(),stage_names.end(),names[en]);if(it!=stage_names.end())stage=it-stage_names.begin();Record rr;rr.fn=en;rr.stage=stage;rr.start=n;rr.count=0;rr.ret=r[1];rr.hist.resize(no);copy(r+10,r+18,rr.args.begin());rec.push_back(rr);stack.push_back(rec.size()-1);}
   counts[ix]++;if(stage>=0)stagecounts[stage][ix]++;rec[stack.back()].hist[i.op]++;n++;U next=pc+i.len;U a=r[i.a],b=r[i.b],c=r[i.c],imm=i.imm;bool branch=false;
   switch(dispatch[i.op]){
    case LI:r[i.a]=imm;break;case LUI:r[i.a]=(S)(int32_t)(imm<<12);break;case AUIPC:r[i.a]=pc+(S)(int32_t)(imm<<12);break;
    case MV:r[i.a]=b;break;case SEXT:r[i.a]=(S)(int32_t)b;break;case ZEXT:r[i.a]=b&255;break;
    case LB:r[i.a]=loadmem(b+imm,1,true);break;case LW:r[i.a]=loadmem(b+imm,4,true);break;case LD:r[i.a]=loadmem(b+imm,8);break;
    case SB:savemem(b+imm,a,1);break;case SW:savemem(b+imm,a,4);break;case SD:savemem(b+imm,a,8);break;
    case ADDI:r[i.a]=b+imm;break;case ADDIW:r[i.a]=(S)(int32_t)(b+imm);break;case SLLI:r[i.a]=b<<(imm&63);break;case SRLI:r[i.a]=b>>(imm&63);break;case SRAI:r[i.a]=(S)b>>(imm&63);break;case ANDI:r[i.a]=b&imm;break;
    case ADD:r[i.a]=b+c;break;case ADDW:r[i.a]=(S)(int32_t)(b+c);break;case SUB:r[i.a]=b-c;break;case SUBW:r[i.a]=(S)(int32_t)(b-c);break;
    case MUL:r[i.a]=b*c;break;case MULW:r[i.a]=(S)(int32_t)(b*c);break;case SLL:r[i.a]=b<<(c&63);break;case SRA:r[i.a]=(S)b>>(c&63);break;case AND:r[i.a]=b&c;break;case SLT:r[i.a]=(S)b<(S)c;break;
    case DIV:r[i.a]=!c?~U(0):((b==U(1)<<63&&c==~U(0))?b:U((S)b/(S)c));break;case DIVU:r[i.a]=c?b/c:~U(0);break;
    case BEQ:branch=a==b;break;case BNE:branch=a!=b;break;case BGE:branch=(S)a>=(S)b;break;case BGEU:branch=a>=b;break;case BLTU:branch=a<b;break;
    case BEQZ:branch=a==0;break;case BNEZ:branch=a!=0;break;case BGEZ:branch=(S)a>=0;break;
    case JAL:r[1]=next;next=imm;break;case J:next=imm;break;case RET:next=r[1];break;
   }
   if(branch){next=imm;taken[ix]++;}r[0]=0;
   if(dispatch[i.op]==RET){while(!stack.empty()&&rec[stack.back()].ret==next){auto&rr=rec[stack.back()];rr.count=n-rr.start;if(find(stage_names.begin(),stage_names.end(),names[rr.fn])!=stage_names.end())stage=-1;stack.pop_back();}}
   pc=next;if(n>60000000)throw runtime_error("runaway");
  }
  int mismatches=0;for(int k=0;k<4096;k++)if(loadmem(output+4*k,4)!=loadmem(golden+cas*0x15000+4*k,4))mismatches++;
  ofstream out("analysis_work/case"+to_string(cas)+".json");out<<"{\"case\":"<<cas<<",\"total\":"<<n<<",\"mismatches\":"<<mismatches<<",\"pc_counts\":[";for(int k=0;k<ni;k++){if(k)out<<',';out<<counts[k];}out<<"],\"branch_taken\":[";for(int k=0;k<ni;k++){if(k)out<<',';out<<taken[k];}out<<"],\"calls\":[";
  for(int k=0;k<(int)rec.size();k++){if(k)out<<',';auto&rr=rec[k];out<<"{\"function\":\""<<names[rr.fn]<<"\",\"stage\":"<<rr.stage<<",\"start\":"<<rr.start<<",\"instructions\":"<<rr.count<<",\"args\":[";for(int j=0;j<8;j++){if(j)out<<',';out<<rr.args[j];}out<<"],\"exclusive_histogram\":[";for(int j=0;j<no;j++){if(j)out<<',';out<<rr.hist[j];}out<<"]}";}out<<"],\"stage_pc_counts\":[";for(int j=0;j<7;j++){if(j)out<<',';out<<'[';for(int k=0;k<ni;k++){if(k)out<<',';out<<stagecounts[j][k];}out<<']';}out<<"]}";
  cerr<<"Case "<<cas<<": "<<n<<" instructions, "<<mismatches<<" output mismatches\n";
  if(mismatches)return 2;
 }
}
