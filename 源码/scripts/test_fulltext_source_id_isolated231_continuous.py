"""Offline endpoint, version-preservation, mutation-scope and hash checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pymilvus import DataType

from benchmark_fulltext_source_id_isolated231_continuous import (
    COLLECTION_PREFIX, DEFAULT_BUCKETS, EXPECTED_SOURCE_COUNT, FIELD,
    SOURCE_COLLECTION, SOURCE_URI, TARGET_URI, Experiment, RestrictedTarget,
    duplicate_counts, patch_rows, validate_destination,
)
from fulltext_source_id_safe_common import IntegrityError, original_hash


DOC = 'a'*32
FIELDS = ['id','doc_id','text','text_embed']


def row(suffix='0'):
    return {'id':DOC+'_'+suffix,'doc_id':DOC,'text':'original text',
        'text_embed':[1.25,-2.5,0.125],FIELD:None}


class CountsOnlySource:
    def __init__(self, counts):
        self.counts=counts
        self.calls=[]
    def query(self, collection, filter, output_fields, **kwargs):
        self.calls.append((collection,filter,output_fields))
        assert collection==SOURCE_COLLECTION and output_fields==['count(*)']
        ids=json.loads(filter[len('id in '):])
        return [{'count(*)':sum(self.counts[key] for key in ids)}]


class InsertOnlyTarget:
    def __init__(self):
        self.inserted=[]
        self.existing=[]
    def list_collections(self):return list(self.existing)
    def create_collection(self, collection, schema, **kwargs):
        self.existing.append(collection)
    def insert(self, collection, data, **kwargs):
        self.inserted.extend(copy.deepcopy(data))
        return {'insert_count':len(data)}


class DestinationTests(unittest.TestCase):
    def test_only_approved_isolated_endpoint_and_new_prefix_are_allowed(self):
        validate_destination(SOURCE_URI,TARGET_URI,COLLECTION_PREFIX+'test')
        for source,target,name in [
            (SOURCE_URI,SOURCE_URI,COLLECTION_PREFIX+'test'),
            (SOURCE_URI,'http://192.0.2.14:19539',COLLECTION_PREFIX+'test'),
            ('http://192.0.2.16:19533',TARGET_URI,COLLECTION_PREFIX+'test'),
            (SOURCE_URI,TARGET_URI,SOURCE_COLLECTION),
            (SOURCE_URI,TARGET_URI+'/',COLLECTION_PREFIX+'test'),
        ]:
            with self.subTest(source=source,target=target,name=name), self.assertRaises(ValueError):
                validate_destination(source,target,name)

    def test_existing_collection_and_other_mutation_names_are_rejected(self):
        raw=InsertOnlyTarget()
        name=COLLECTION_PREFIX+'test'
        writer=RestrictedTarget(raw,name)
        raw.existing.append(name)
        with self.assertRaises(IntegrityError):writer.create(object())
        with self.assertRaises(IntegrityError):writer.insert(SOURCE_COLLECTION,[row()])
        with self.assertRaises(IntegrityError):writer.insert(name,[row()]*201)
        self.assertEqual(raw.inserted,[])
        self.assertFalse(hasattr(writer,'delete'))
        self.assertFalse(hasattr(writer,'drop_collection'))
        self.assertFalse(hasattr(writer,'upsert'))

    def test_constructor_is_offline_bounded_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new'
            with patch('benchmark_fulltext_source_id_isolated231_continuous.MilvusClient',side_effect=AssertionError('network forbidden')):
                experiment=Experiment(root,COLLECTION_PREFIX+'test')
                experiment.db.close()
            with self.assertRaises(RuntimeError):Experiment(root,COLLECTION_PREFIX+'test')
            for rows,seconds in [(199999,900),(800001,900),(600000,599),(600000,901)]:
                with self.subTest(rows=rows,seconds=seconds),self.assertRaises(ValueError):
                    Experiment(Path(directory)/'invalid',COLLECTION_PREFIX+'test',rows,seconds)
        self.assertEqual(len(DEFAULT_BUCKETS),30)
        self.assertEqual(len(set(DEFAULT_BUCKETS)),30)


class VersionAndHashTests(unittest.TestCase):
    def test_duplicate_guard_finds_each_physical_duplicate_without_vectors(self):
        ids=['key1','key2','key3','key4']
        source=CountsOnlySource(dict(zip(ids,[1,3,1,2])))
        self.assertEqual(duplicate_counts(source,ids),{'key2':3,'key4':2})
        self.assertTrue(all(fields==['count(*)'] for _,_,fields in source.calls))

    def test_disappeared_or_repeated_primary_keys_fail_closed(self):
        with self.assertRaises(IntegrityError):duplicate_counts(CountsOnlySource({'key':0}),['key'])
        with self.assertRaises(IntegrityError):duplicate_counts(CountsOnlySource({'key':2}),['key','key'])

    def test_patching_preserves_original_text_vector_and_each_returned_row(self):
        rows=[row(),row()]
        before=copy.deepcopy(rows)
        audit,added,missing=patch_rows(rows,{DOC:['source-id']},FIELDS)
        self.assertEqual((len(rows),len(audit),added,missing),(2,2,2,0))
        for old,new in zip(before,rows):
            self.assertEqual(original_hash(old,FIELDS),original_hash(new,FIELDS))
            self.assertEqual(new[FIELD],['source-id'])
            self.assertEqual({k:v for k,v in old.items() if k!=FIELD},
                {k:v for k,v in new.items() if k!=FIELD})

    def test_conflicting_source_id_fails_and_missing_source_does_not_erase_value(self):
        existing=row();existing[FIELD]=['previous']
        with self.assertRaises(IntegrityError):patch_rows([existing],{DOC:['different']},FIELDS)
        patch_rows([existing],{},FIELDS)
        self.assertEqual(existing[FIELD],['previous'])


class CopyBatchTests(unittest.TestCase):
    def create(self,directory,skip):
        experiment=Experiment(Path(directory)/'new',COLLECTION_PREFIX+'test',skip_duplicate_keys=skip)
        experiment.original_fields=FIELDS
        experiment.db.execute("INSERT INTO batches(seq,phase,status,rows) VALUES(1,'test','queued',2)")
        experiment.db.commit()
        raw=InsertOnlyTarget()
        source=CountsOnlySource({DOC+'_0':2,DOC+'_1':1})
        experiment.worker=lambda:(source,RestrictedTarget(raw,experiment.collection),object())
        experiment.sources=lambda ids,session:{DOC:['source-id']}
        return experiment,raw

    def test_default_duplicate_policy_aborts_before_any_insert_and_records_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment,raw=self.create(directory,False)
            try:
                with self.assertRaises(IntegrityError):experiment.copy_batch(1,'test',[row('0'),row('1')],0)
                self.assertEqual(raw.inserted,[])
                details=json.loads((experiment.root/'excluded_duplicate_keys.json').read_text())
                self.assertEqual(details,[{'id':DOC+'_0','physical_count':2}])
            finally:experiment.db.close()

    def test_explicit_benchmark_skip_excludes_whole_key_but_never_alters_source(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment,raw=self.create(directory,True)
            originals=[row('0'),row('1')]
            before=copy.deepcopy(originals)
            try:
                result=experiment.copy_batch(1,'test',copy.deepcopy(originals),0.3)
                self.assertEqual(result['rows_inserted'],1)
                self.assertFalse(result['readback_verified'])
                self.assertEqual([entry['id'] for entry in raw.inserted],[DOC+'_1'])
                self.assertEqual(originals,before)
                self.assertEqual(experiment.report['excluded_duplicate_keys'],1)
                self.assertEqual(experiment.report['excluded_physical_versions'],2)
                self.assertIn('cannot prove a lossless full migration',experiment.report['integrity_boundary'])
                stored=experiment.db.execute('SELECT original_hash,source_field_before,source_field_after FROM expected_rows').fetchone()
                self.assertEqual(stored,(original_hash(before[1],FIELDS),'null','["source-id"]'))
            finally:experiment.db.close()

    def test_original_schema_fields_are_copied_and_source_array_can_be_added(self):
        class Schema:
            def __init__(self):self.fields=[]
            def add_field(self,**kwargs):self.fields.append(kwargs)
        class Target:
            def create_schema(self,**kwargs):return Schema()
        experiment=Experiment.__new__(Experiment)
        experiment.target=Target()
        source={'auto_id':False,'enable_dynamic_field':False,'fields':[
            {'name':'id','type':DataType.VARCHAR,'is_primary':True,'params':{'max_length':100}},
            {'name':'text_embed','type':DataType.FLOAT_VECTOR,'params':{'dim':2560}}]}
        schema=experiment.clone_schema(source)
        self.assertEqual(schema.fields[0]['field_name'],'id')
        self.assertTrue(schema.fields[0]['is_primary'])
        self.assertEqual(schema.fields[1]['dim'],2560)
        self.assertEqual(schema.fields[2]['field_name'],FIELD)
        self.assertEqual(schema.fields[2]['datatype'],DataType.ARRAY)
        self.assertEqual(schema.fields[2]['element_type'],DataType.VARCHAR)
        self.assertTrue(schema.fields[2]['nullable'])


class StreamVerificationTests(unittest.TestCase):
    def test_stop_marker_drains_copy_without_starting_new_index_or_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment=Experiment(Path(directory)/'new',COLLECTION_PREFIX+'stoptest')
            experiment.preflight=lambda:None
            def stop_copy(*args):
                experiment.report['phases'].append({'elapsed_seconds_including_inflight_waits':1})
                (experiment.root/'STOP_REQUESTED').touch()
            experiment.copy_phase=stop_copy
            experiment.verify=lambda:(_ for _ in ()).throw(AssertionError('verification must not start after stop'))
            experiment.progress=experiment.save
            experiment.run()
            report=json.loads((experiment.root/'report.json').read_text())
            self.assertEqual(report['status'],'stopped_by_user_test_collection_preserved_unverified')
            self.assertEqual(len(report['phases']),1)

    def run_verification(self,directory,corrupt=False):
        experiment=Experiment(Path(directory)/'new',COLLECTION_PREFIX+'verifytest')
        original=[row('0'),row('1')]
        patched=copy.deepcopy(original)
        audit,_,_=patch_rows(patched,{DOC:['source-id']},FIELDS)
        for number,(key,digest,before,after) in enumerate(audit):
            experiment.db.execute('INSERT INTO expected_rows(seq,row_no,id,original_hash,source_field_before,source_field_after) VALUES(1,?,?,?,?,?)',
                (number,key,digest,before,after))
        experiment.db.execute('INSERT INTO batches(seq,phase,status,rows,inserted,result) VALUES(1,?,?,?,?,?)',
            ('test','inserted_pending_verification',2,2,json.dumps({'readback_verified':False})))
        experiment.db.commit()
        experiment.original_fields=FIELDS
        experiment.source_fields=FIELDS+[FIELD]
        experiment.source_schema={'original_schema':True}
        experiment.target_collections_before={'old_untouched'}
        experiment.report['source_count_before']=EXPECTED_SOURCE_COUNT
        experiment.report['source_field_count_before']=0
        experiment.report['inserted_rows_pending_verification']=2
        experiment.report['phases']=[{'readback_verified':False}]
        class Iterator:
            def __init__(self,rows):self.rows=rows
            def next(self):
                rows=self.rows
                self.rows=[]
                return copy.deepcopy(rows)
            def close(self):pass
        class Indexes:
            def add_index(self,**kwargs):
                assert kwargs['index_type']=='FLAT'
        class Source:
            def get(self,collection,ids,**kwargs):
                assert collection==SOURCE_COLLECTION and len(ids)<=200
                return [copy.deepcopy(entry) for entry in original if entry['id'] in ids]
            def describe_collection(self,collection):
                assert collection==SOURCE_COLLECTION
                return experiment.source_schema
        class Target:
            def flush(self,collection,**kwargs):assert collection==experiment.collection
            def prepare_index_params(self):return Indexes()
            def create_index(self,collection,**kwargs):assert collection==experiment.collection
            def load_collection(self,collection,**kwargs):assert collection==experiment.collection
            def query_iterator(self,collection,batch_size,**kwargs):
                assert collection==experiment.collection and batch_size<=200
                return Iterator(patched)
            def list_collections(self):return ['old_untouched',experiment.collection]
        experiment.source=Source()
        experiment.target=Target()
        experiment.count=lambda client,collection,expression='':(
            0 if expression else EXPECTED_SOURCE_COUNT) if collection==SOURCE_COLLECTION else 2
        if corrupt:
            patched[0]['text_embed'][0]=9.0
        return experiment

    def test_full_stream_validates_every_original_vector_and_records_batch_success(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment=self.run_verification(directory)
            try:
                experiment.verify()
                self.assertEqual(experiment.report['verified_rows'],2)
                self.assertEqual(experiment.report['source_sample_rows_unchanged'],2)
                self.assertTrue(experiment.report['all_copied_original_fields_vectors_and_source_ids_verified'])
                status,result=experiment.db.execute('SELECT status,result FROM batches').fetchone()
                self.assertEqual(status,'completed_and_verified')
                self.assertTrue(json.loads(result)['readback_verified'])
                self.assertFalse(experiment.report['production_ready'])
            finally:experiment.db.close()

    def test_changed_target_vector_fails_readback_without_any_production_write(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment=self.run_verification(directory,True)
            try:
                with self.assertRaises(IntegrityError):experiment.verify()
                self.assertEqual(experiment.report['production_writes'],0)
                self.assertFalse(experiment.report.get('all_copied_original_fields_vectors_and_source_ids_verified',False))
            finally:experiment.db.close()


if __name__=='__main__':unittest.main()
