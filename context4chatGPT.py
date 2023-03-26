from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
import os
import sys
import click


def check_api_key():
    '''Check if the API key is set in the environment variables.
    The key is not explicitly used in this script, but the environment variable
    is required for the llama_index library to work.'''
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        click.echo(f"API key is: {api_key}")
    else:
        click.echo("API key is not set. Use: export OPENAI_API_KEY=your_key")
        sys.exit(1)
    return api_key


@click.command(help='''Uses the file in INPUT_FOLDER to create context
                     for chatGPT, takes questions from user and passes
                     them to chatGPT.''')
@click.argument('input_folder', type=click.Path(exists=True))
@click.option('--recreate_index', '-r',
              is_flag=True, show_default=True,
              default=False,
              help='recreate the index from scratch')
def main(input_folder, recreate_index):
    question = ' '
    check_api_key()
    if recreate_index or not os.path.exists('index.json'):
        loader = SimpleDirectoryReader(input_folder)
        documents = loader.load_data()
        index = GPTSimpleVectorIndex(documents)
        index.save_to_disk('index.json')
        click.echo(
            f'Successfully processed {len(os.listdir(input_folder))} files.')
    else:
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while question:
        question = input("Type a question :")
        if question != '':
            response = index.query(question)
            print(response)
        else:
            break


if __name__ == "__main__":
    main()
