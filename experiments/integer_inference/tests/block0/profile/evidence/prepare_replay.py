from pathlib import Path
import struct,re,json
root=Path(__file__).resolve().parents[1]
src=root/'analysis_input/latest/build/block0/profile'
e=(src/'test_block0_profile.elf').read_bytes()
h=struct.unpack_from('<16sHHIQQQIHHHHHH',e)
mem=bytearray(16*1024*1024)
for i in range(h[10]):
 p=struct.unpack_from('<IIQQQQQQ',e,h[5]+i*h[9])
 if p[0]==1:
  dest=p[3]-0x80000000
  assert 0<=dest and dest+p[6]<=len(mem)
  mem[dest:dest+p[5]]=e[p[2]:p[2]+p[5]]
(root/'analysis_work/memory.bin').write_bytes(mem)
regs={'zero':0,'ra':1,'sp':2,'gp':3,'tp':4,'t0':5,'t1':6,'t2':7,'s0':8,'fp':8,'s1':9,**{'a'+str(i):10+i for i in range(8)},**{'s'+str(i):16+i for i in range(2,12)},'t3':28,'t4':29,'t5':30,'t6':31}
ops=[];fn=[];ins=[]
for line in (src/'test_block0_profile.dis').read_text().splitlines():
 m=re.match(r'([0-9a-f]+) <([^>]+)>:',line)
 if m and 0x80000136<=int(m[1],16)<0x80001226:
  fn.append((int(m[1],16),m[2]))
 m=re.match(r'\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*)',line)
 if not m or not 0x80000136<=int(m[1],16)<0x80001226:continue
 addr,encoding,op,oper=m.groups();addr=int(addr,16);oper=oper.split('#')[0].strip()
 assert mem[addr-0x80000000:addr-0x80000000+len(encoding)//2] == bytes.fromhex(encoding)[::-1], hex(addr)
 if op not in ops:ops.append(op)
 parts=[p.strip() for p in oper.split(',') if p.strip()]
 a=b=c=imm=0
 if op in ['li','lui','auipc']:a=regs[parts[0]];imm=int(parts[1],0)
 elif op in ['mv','sext.w','zext.b']:a,b=map(regs.get,parts)
 elif op in ['lb','lw','ld','sb','sw','sd']:
  a=regs[parts[0]];mm=re.fullmatch(r'(-?(?:0x)?[0-9a-f]+)\((\w+)\)',parts[1]);imm=int(mm[1],0);b=regs[mm[2]]
 elif op in ['addi','addiw','slli','srli','srai','andi']:a=regs[parts[0]];b=regs[parts[1]];imm=int(parts[2],0)
 elif op in ['add','addw','sub','subw','mul','mulw','sll','sra','and','slt','div','divu']:a,b,c=map(regs.get,parts)
 elif op in ['beq','bne','bge','bgeu','bltu']:a=regs[parts[0]];b=regs[parts[1]];imm=int(parts[2].split()[0],16)
 elif op in ['beqz','bnez','bgez']:a=regs[parts[0]];imm=int(parts[1].split()[0],16)
 elif op in ['jal','j']:imm=int(parts[0].split()[0],16)
 elif op!='ret':raise ValueError((op,parts))
 owner=max(i for i,f in enumerate(fn) if f[0]<=addr)
 ins.append((addr,len(encoding)//2,ops.index(op),a,b,c,imm,owner))
lines=[f'{len(ops)} {len(fn)} {len(ins)}']+ops+[f'{a} {n}' for a,n in fn]+[' '.join(map(str,r)) for r in ins]
(root/'analysis_work/program.txt').write_text('\n'.join(lines)+'\n')
(root/'analysis_work/program_metadata.json').write_text(json.dumps({'ops':ops,'functions':fn,'instructions':ins},indent=2))
print('Decoded',len(ins),'instructions;',len(fn),'functions;',len(ops),'mnemonics')
