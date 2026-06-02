#include "MambaAraController.hpp"
#include "core/Platform.hpp"

// -----------------------------------------------------------------------------
// PlaybackRenderer
// -----------------------------------------------------------------------------

MambaPlaybackRenderer::MambaPlaybackRenderer(ARA::PlugIn::DocumentController* documentController)
: ARA::PlugIn::PlaybackRenderer(documentController)
{
    MambaInferenceEngine::Config config;
    config.modelPath = mambaResolveModulePath((void*)&ARAGetFactory, "mamba_rt.onnx");
    config.maxBlockSize = 1024;
    rtEngine_ = std::make_unique<MambaInferenceEngine>(config);
    rtEngine_->init();
}

MambaPlaybackRenderer::~MambaPlaybackRenderer() noexcept = default;

// -----------------------------------------------------------------------------
// DocumentController
// -----------------------------------------------------------------------------

ARA::PlugIn::Document* MambaDocumentController::doCreateDocument() noexcept
{
    return new ARA::PlugIn::Document(this);
}

void MambaDocumentController::doDestroyDocument(ARA::PlugIn::Document* document) noexcept
{
    delete document;
}

ARA::PlugIn::MusicalContext* MambaDocumentController::doCreateMusicalContext(ARA::PlugIn::Document* document, ARA::ARAMusicalContextHostRef hostRef) noexcept
{
    return new ARA::PlugIn::MusicalContext(document, hostRef);
}

void MambaDocumentController::doDestroyMusicalContext(ARA::PlugIn::MusicalContext* musicalContext) noexcept
{
    delete musicalContext;
}

void MambaDocumentController::doUpdateMusicalContextContent(ARA::PlugIn::MusicalContext* context, const ARA::ARAContentTimeRange* range, ARA::ContentUpdateScopes scopeFlags) noexcept
{
    // Musical context (tempo, time signature) changed.
    // TODO: When model conditions on tempo/key, extract context and call
    // engine->setConditioning() with derived features here.
    // For now, mark content as dirty and continue.
    contentDirty_.store(true, std::memory_order_release);
    (void)context; (void)range; (void)scopeFlags;
}

ARA::PlugIn::RegionSequence* MambaDocumentController::doCreateRegionSequence(ARA::PlugIn::Document* document, ARA::ARARegionSequenceHostRef hostRef) noexcept
{
    return new ARA::PlugIn::RegionSequence(document, hostRef);
}

void MambaDocumentController::doDestroyRegionSequence(ARA::PlugIn::RegionSequence* regionSequence) noexcept
{
    delete regionSequence;
}

ARA::PlugIn::AudioSource* MambaDocumentController::doCreateAudioSource(ARA::PlugIn::Document* document, ARA::ARAAudioSourceHostRef hostRef) noexcept
{
    return new ARA::PlugIn::AudioSource(document, hostRef);
}

void MambaDocumentController::doDestroyAudioSource(ARA::PlugIn::AudioSource* audioSource) noexcept
{
    delete audioSource;
}

void MambaDocumentController::doUpdateAudioSourceContent(ARA::PlugIn::AudioSource* source, const ARA::ARAContentTimeRange* range, ARA::ContentUpdateScopes scopeFlags) noexcept
{
    // Audio source content (waveform edits, comping) changed.
    // TODO: When analysis model is loaded, trigger re-analysis here.
    // For now, mark content dirty and continue.
    contentDirty_.store(true, std::memory_order_release);
    (void)source; (void)range; (void)scopeFlags;
}

void MambaDocumentController::willEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource* source, bool enable) noexcept
{
    // When enabled, companion plugin can read source samples via
    // ARA::PlugIn::HostAudioReader for offline analysis.
    // Reset dirty flag since we may re-read content.
    if (enable) {
        contentDirty_.store(false, std::memory_order_release);
    }
    (void)source;
}

void MambaDocumentController::didEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource* source, bool enable) noexcept
{
    // Host has granted/revoked access to audio source sample data.
    // When enabled, companion plugin can read source samples via
    // ARA::PlugIn::HostAudioReader for offline analysis.
    // TODO: If enabled and content is dirty, push AnalysisQueueDrain
    // to the companion plugin's SPSC task queue (requires host bridge).
    // For now, just mark content dirty if re-enabled.
    if (enable) {
        contentDirty_.store(true, std::memory_order_release);
    }
    (void)source;
}

ARA::PlugIn::AudioModification* MambaDocumentController::doCreateAudioModification(ARA::PlugIn::AudioSource* audioSource, ARA::ARAAudioModificationHostRef hostRef, const ARA::PlugIn::AudioModification* optionalModificationToClone) noexcept
{
    return new ARA::PlugIn::AudioModification(audioSource, hostRef, optionalModificationToClone);
}

void MambaDocumentController::doDestroyAudioModification(ARA::PlugIn::AudioModification* audioModification) noexcept
{
    delete audioModification;
}

ARA::PlugIn::PlaybackRegion* MambaDocumentController::doCreatePlaybackRegion(ARA::PlugIn::AudioModification* modification, ARA::ARAPlaybackRegionHostRef hostRef) noexcept
{
    return new ARA::PlugIn::PlaybackRegion(modification, hostRef);
}

void MambaDocumentController::doDestroyPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    delete playbackRegion;
}

ARA::PlugIn::PlaybackRenderer* MambaDocumentController::doCreatePlaybackRenderer() noexcept
{
    return new MambaPlaybackRenderer(this);
}

void MambaDocumentController::doDestroyPlaybackRenderer(ARA::PlugIn::PlaybackRenderer* playbackRenderer) noexcept
{
    delete playbackRenderer;
}

ARA::PlugIn::EditorRenderer* MambaDocumentController::doCreateEditorRenderer() noexcept
{
    // EditorRenderer is deferred until GUI is built (CLAP_EXT_GUI).
    // Returns nullptr — DAW will not show ARA editor views.
    // See CLAP_EXT_GUI tracking in TODO.md.
    return nullptr;
}

void MambaDocumentController::doDestroyEditorRenderer(ARA::PlugIn::EditorRenderer* editorRenderer) noexcept
{
    // EditorRenderer is deferred — this is a no-op stub.
    // If EditorRenderer is created in the future, delete here.
    (void)editorRenderer;
}

ARA::PlugIn::EditorView* MambaDocumentController::doCreateEditorView() noexcept
{
    // EditorView is deferred until GUI is built (CLAP_EXT_GUI).
    // Returns nullptr — DAW will not show ARA editor views.
    // See CLAP_EXT_GUI tracking in TODO.md.
    return nullptr;
}

void MambaDocumentController::doDestroyEditorView(ARA::PlugIn::EditorView* editorView) noexcept
{
    // EditorView is deferred — this is a no-op stub.
    // If EditorView is created in the future, delete here.
    (void)editorView;
}

bool MambaDocumentController::doRestoreObjectsFromArchive(ARA::PlugIn::HostArchiveReader* archiveReader, const ARA::PlugIn::RestoreObjectsFilter*) noexcept
{
    if (!archiveReader) return false;

    struct ArchivedParams {
        float bypass;
        float wetDry;
        float intensity;
    };

    ARA::ARASize archiveSize = archiveReader->getArchiveSize();
    if (archiveSize < sizeof(ArchivedParams)) return false;

    ArchivedParams params{};
    if (!archiveReader->readBytesFromArchive(0, sizeof(params), reinterpret_cast<ARA::ARAByte*>(&params))) {
        return false;
    }

    bypass.store(params.bypass, std::memory_order_release);
    wetDry.store(params.wetDry, std::memory_order_release);
    intensity.store(params.intensity, std::memory_order_release);

    return true;
}

bool MambaDocumentController::doStoreObjectsToArchive(ARA::PlugIn::HostArchiveWriter* archiveWriter, const ARA::PlugIn::StoreObjectsFilter*) noexcept
{
    if (!archiveWriter) return false;

    struct ArchivedParams {
        float bypass;
        float wetDry;
        float intensity;
    };

    ArchivedParams params;
    params.bypass = bypass.load(std::memory_order_acquire);
    params.wetDry = wetDry.load(std::memory_order_acquire);
    params.intensity = intensity.load(std::memory_order_acquire);

    return archiveWriter->writeBytesToArchive(0, sizeof(params), reinterpret_cast<const ARA::ARAByte*>(&params));
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

const char* MambaFactoryConfig::getFactoryID() const noexcept
{
    return "com.mamba.audio.suite.ara";
}

const char* MambaFactoryConfig::getPlugInName() const noexcept
{
    return "Mamba-SSM Processor (ARA)";
}

const char* MambaFactoryConfig::getManufacturerName() const noexcept
{
    return "Elite DSP";
}

const char* MambaFactoryConfig::getInformationURL() const noexcept
{
    return MAMBA_PROJECT_URL;
}

const char* MambaFactoryConfig::getVersion() const noexcept
{
    return "1.1.0";
}

const char* MambaFactoryConfig::getDocumentArchiveID() const noexcept
{
    return "com.mamba.audio.suite.archive";
}

ARA::PlugIn::DocumentController* MambaFactoryConfig::createDocumentController(const ARA::PlugIn::PlugInEntry* entry, const ARA::ARADocumentControllerHostInstance* instance) const noexcept
{
    return new MambaDocumentController(entry, instance);
}

const ARA::ARAFactory* getMambaARAFactory() noexcept
{
    return ARA::PlugIn::PlugInEntry::getPlugInEntry<MambaFactoryConfig, MambaDocumentController>()->getFactory();
}

extern "C" const ARA::ARAFactory* ARAGetFactory()
{
    return getMambaARAFactory();
}
