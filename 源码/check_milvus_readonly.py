from pymilvus import connections, utility, Collection
hosts=['192.0.2.14','192.0.2.15','192.0.2.16']
for host in hosts:
    alias='c_'+host.split('.')[-1]
    print('===',host,'===')
    connections.connect(alias=alias,host=host,port=19533,timeout=8)
    for name in sorted(utility.list_collections(using=alias)):
        c=Collection(name,using=alias)
        pk=next(f.name for f in c.schema.fields if f.is_primary)
        try:
            rows=c.query(expr='',output_fields=[pk],limit=1,consistency_level='Bounded')
            print(name,'entities=',c.num_entities,'state=',utility.load_state(name,using=alias),'query_rows=',len(rows),'sample_pk=',str(rows[0].get(pk)) if rows else '<none>')
        except Exception as e:
            print(name,'entities=',c.num_entities,'state=',utility.load_state(name,using=alias),'QUERY_ERROR=',type(e).__name__,str(e)[:200])
    connections.disconnect(alias)
